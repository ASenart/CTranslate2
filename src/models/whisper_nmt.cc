#include "ctranslate2/models/whisper_nmt.h"

#include <algorithm>
#include <spdlog/spdlog.h>

#include "ctranslate2/decoding.h"

#include "dispatch.h"
#include "dtw.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

      const Vocabulary &WhisperNmtModel::get_vocabulary() const {
        return *_vocabulary;
      }

      size_t WhisperNmtModel::num_source_vocabularies() const {
        return _source_vocabularies.size();
      }

      const Vocabulary &WhisperNmtModel::get_source_vocabulary(size_t index) const {
        return *_source_vocabularies.at(index);
      }

      const Vocabulary &WhisperNmtModel::get_target_vocabulary() const {
        return *_target_vocabulary;
      }

      size_t WhisperNmtModel::current_spec_revision() const {
        return 7;
      }

      void WhisperNmtModel::load_vocabularies(ModelReader &model_reader) {
        {
          VocabularyInfo vocab_info;
          vocab_info.unk_token = config["unk_token"];
          vocab_info.bos_token = config["bos_token"];
          vocab_info.eos_token = config["eos_token"];

          auto shared_vocabulary = load_vocabulary(model_reader, "shared_vocabulary", vocab_info);

          if (shared_vocabulary) {
            _target_vocabulary = shared_vocabulary;
            _source_vocabularies = {shared_vocabulary};

          } else {
            _target_vocabulary = load_vocabulary(model_reader, "target_vocabulary", vocab_info);
            if (!_target_vocabulary)
              throw std::runtime_error("Cannot load the target vocabulary from the model directory");

            auto source_vocabulary = load_vocabulary(model_reader, "source_vocabulary", vocab_info);

            if (source_vocabulary) {
              _source_vocabularies = {source_vocabulary};
            } else {
              for (size_t i = 1;; i++) {
                const std::string filename = "source_" + std::to_string(i) + "_vocabulary";
                auto vocabulary = load_vocabulary(model_reader, filename, vocab_info);

                if (!vocabulary)
                  break;

                _source_vocabularies.emplace_back(vocabulary);
              }
            }

            if (_source_vocabularies.empty())
              throw std::runtime_error("Cannot load the source vocabulary from the model directory");
          }
        }
      }

      void WhisperNmtModel::initialize(ModelReader &model_reader) {
        //VocabularyInfo vocab_info;
        //vocab_info.unk_token = "<|endoftext|>";
        //vocab_info.bos_token = "<|startoftranscript|>";
        //vocab_info.eos_token = "<|endoftext|>";

        //_vocabulary = load_vocabulary(model_reader, "shared_vocabulary", std::move(vocab_info));
        //if (!_vocabulary)
        //  throw std::runtime_error("Cannot load the vocabulary from the model directory");
        // load vocabularies for nmt encoder and decoder
        load_vocabularies(model_reader);
      }

      bool WhisperNmtModel::is_quantizable(const std::string &variable_name) const {
        return Model::is_quantizable(variable_name);
      }

      bool WhisperNmtModel::is_linear_weight(const std::string &variable_name) const {
        return is_quantizable(variable_name) && variable_name.find("embeddings") == std::string::npos;
      }

      std::unique_ptr<Model> WhisperNmtModel::clone() const {
        return std::make_unique<WhisperNmtModel>(*this);
      }


      std::unique_ptr<WhisperNmtReplica> WhisperNmtReplica::create_from_model(const Model &model) {
        if (!dynamic_cast<const WhisperNmtModel *>(&model))
          throw std::invalid_argument("The model is not a Whisper model");

        const auto scoped_device_setter = model.get_scoped_device_setter();
        const auto model_ptr = model.shared_from_this();
        const auto concrete_model = std::static_pointer_cast<const WhisperNmtModel>(model_ptr);
        return std::make_unique<WhisperNmtReplica>(concrete_model);
      }

      WhisperNmtReplica::WhisperNmtReplica(const std::shared_ptr<const WhisperNmtModel> &model)
              : ModelReplica(model), _model(model),
                _encoder(std::make_unique<layers::WhisperEncoder>(*model, "whisper_encoder")),
                _transformer_encoder(std::make_unique<layers::TransformerEncoder>(*model, "transformer_encoder")),
                _transformer_decoder(std::make_unique<layers::TransformerDecoder>(*model, "transformer_decoder")),
                _connector(std::make_unique<layers::WhisperConnector>(*model, "connector")) {
        const auto &vocabulary = model->get_source_vocabulary(0);
        _sot_id = vocabulary.bos_id();
        _eot_id = vocabulary.eos_id();
        _no_timestamps_id = vocabulary.to_id("<|notimestamps|>");
        _no_speech_id = vocabulary.to_id("<|nospeech|>");
        if (_no_speech_id == vocabulary.unk_id())
          _no_speech_id = vocabulary.to_id("<|nocaptions|>");
        _is_multilingual = vocabulary.size() >= 51865;
        _n_mels = _encoder->input_size();
        _num_languages = vocabulary.size() - 51765 - (_is_multilingual ? 1 : 0);
      }

      StorageView WhisperNmtReplica::encode(StorageView features, const bool to_cpu) {
        PROFILE("WhisperReplica::encode");

#ifdef CT2_WITH_CUDA
        const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

        const auto scoped_device_setter = _model->get_scoped_device_setter();
        const Device device = _model->device();
        const DataType dtype = _encoder->output_type();
        features.move_to(device, dtype);

        StorageView encoder_output(dtype, device);
        (*_encoder)(features, encoder_output);

        if (to_cpu) {
          if (device != Device::CPU)
            encoder_output = encoder_output.to(Device::CPU);
          return encoder_output;
        }

        // Ensure all operations are finished before returning the output.
        synchronize_stream(device);

        return encoder_output;
      }

      StorageView WhisperNmtReplica::maybe_encode(StorageView features) {
        const Device device = _model->device();
        const DataType dtype = _encoder->output_type();

        features.move_to(device, dtype);

        if (_encoder->is_encoded(features))
          return features;

        StorageView encoder_output(dtype, device);
        (*_encoder)(features, encoder_output);
        return encoder_output;
      }


      static std::vector<float> get_no_speech_probs_from_logits(const StorageView &logits,
                                                                const size_t no_speech_id) {
        const Device device = logits.device();
        const DataType dtype = logits.dtype();

        StorageView probs(dtype, device);
        ops::SoftMax()(logits, probs);

        StorageView gather_ids({probs.dim(0)}, int32_t(no_speech_id), device);
        StorageView no_speech_probs(dtype, device);
        ops::Gather(/*axis=*/1, /*batch_dims=*/1)(probs, gather_ids, no_speech_probs);

        if (no_speech_probs.dtype() != DataType::FLOAT32)
          no_speech_probs = no_speech_probs.to_float32();
        return no_speech_probs.to_vector<float>();
      }

      static size_t get_sot_index(const std::vector<size_t> &prompt, const size_t sot_id) {
        const auto sot_it = std::find(prompt.begin(), prompt.end(), sot_id);
        if (sot_it == prompt.end())
          throw std::invalid_argument("<|startoftranscript|> token was not found in the prompt");

        return std::distance(prompt.begin(), sot_it);
      }

      static size_t get_prompt_length(const std::vector<size_t> &prompt,
                                      const size_t sot_id,
                                      const size_t no_timestamps_id) {
        size_t index = get_sot_index(prompt, sot_id);
        while (index < prompt.size() && prompt[index] >= sot_id && prompt[index] <= no_timestamps_id)
          index++;
        return index;
      }

      static void check_prompts(const std::vector<std::vector<size_t>> &prompts,
                                const size_t sot_id,
                                const size_t no_timestamps_id,
                                size_t &sot_index,
                                size_t &prompt_length) {
        bool first = true;

        for (const auto &prompt: prompts) {
          const auto batch_sot_index = get_sot_index(prompt, sot_id);
          const auto batch_prompt_length = get_prompt_length(prompt, sot_id, no_timestamps_id);

          if (first) {
            sot_index = batch_sot_index;
            prompt_length = batch_prompt_length;
          } else if (batch_sot_index != sot_index) {
            throw std::invalid_argument("The generate method currently requires the "
                                        "<|startoftranscript|> token to be at the same position "
                                        "in all batches. To work around this limitation, "
                                        "simply adapt the number of previous text tokens in each "
                                        "batch.");
          } else if (batch_prompt_length != prompt_length) {
            throw std::invalid_argument("The generate method currently requires each batch to have "
                                        "the same number of task tokens after <|startoftranscript|>.");
          }

          first = false;
        }
      }

      class ApplyTimestampRules;

      class GetNoSpeechProbs : public LogitsProcessor {
      private:
          const size_t _no_speech_id;
          std::vector<float> _no_speech_probs;

      public:
          GetNoSpeechProbs(const size_t no_speech_id)
                  : _no_speech_id(no_speech_id) {
          }

          const std::vector<float> &get_no_speech_probs() const {
            return _no_speech_probs;
          }

          bool apply_first() const override {
            return true;
          }

          void apply(dim_t step,
                     StorageView &logits,
                     DisableTokens &,
                     const StorageView &,
                     const std::vector<dim_t> &batch_offset,
                     const std::vector<std::vector<size_t>> *) override {
            if (step == 0) {
              const auto no_speech_probs = get_no_speech_probs_from_logits(logits, _no_speech_id);

              const size_t batch_size = batch_offset.size();
              const size_t beam_size = logits.dim(0) / batch_size;

              _no_speech_probs.reserve(batch_size);
              for (size_t i = 0; i < batch_size; ++i)
                _no_speech_probs.emplace_back(no_speech_probs[i * beam_size]);
            }
          }
      };

      std::vector<WhisperNmtTranslationResult>
      WhisperNmtReplica::translate(StorageView features,
                                   const std::vector<std::vector<std::string>>& source_prefix,
                                   const std::vector<std::vector<std::string>>& target_prefix,
                                   const WhisperNmtOptions &options) {
        PROFILE("WhisperReplica::translate");

#ifdef CT2_WITH_CUDA
        const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif
        auto target = target_prefix;
        if (target.empty())
          target.resize(features.shape()[0]);

        auto source = source_prefix;
        if (source.empty())
          source.resize(features.shape()[0]);

        const auto &vocabulary = _model->get_vocabulary();
        const auto scoped_device_setter = _model->get_scoped_device_setter();

        StorageView encode_features = maybe_encode(std::move(features));

        // MLP
        StorageView mlp_output(_encoder->output_type(), _model->device());
        (*_connector)(encode_features, mlp_output);
        // NMT
        std::vector<WhisperNmtTranslationResult> final_results = _run_translation(mlp_output,
                                                                                  source,
                                                                                  target,
                                                                                  options);

        return final_results;
      }

      void
      WhisperNmtReplica::_nmt_encode(StorageView &features_ids,
                                     const std::vector<std::vector<size_t>>& sources_prefix,
                                     const std::vector<std::vector<size_t>>& sources_suffix,
                                     StorageView &memory,
                                     StorageView &memory_lengths) {

        std::vector<StorageView> source_features_prefix;
        std::vector<StorageView> source_features_suffix;
        source_features_prefix.emplace_back(layers::make_sequence_inputs(sources_prefix,
                                                                  _model->device(),
                                                                  _model->preferred_size_multiple(),
                                                                  &memory_lengths));
        source_features_suffix.emplace_back(layers::make_sequence_inputs(sources_suffix,
                                                                         _model->device(),
                                                                         _model->preferred_size_multiple(),
                                                                         nullptr));
        auto batch_size = sources_prefix.size();
        for (size_t i = 0; i < batch_size; i++) {
          if (memory_lengths) {
            memory_lengths.at<int32_t>(i) = source_features_prefix[0].shape()[1]
                                                  + features_ids.shape()[1]
                                                  + source_features_suffix[0].shape()[1];
          }
        }

        (*_transformer_encoder)(features_ids, source_features_prefix, source_features_suffix, &memory_lengths, memory);
      }

      std::vector<std::vector<size_t>>
      WhisperNmtReplica::make_source_prefix_ids(const std::vector<std::vector<std::string>>& source) const {
        const auto &target_vocabulary = _model->get_source_vocabulary(0);
        return target_vocabulary.to_ids(source, 0, nullptr, nullptr);
      }

      std::vector<std::vector<size_t>>
      WhisperNmtReplica::make_target_prefix_ids(const std::vector<std::vector<std::string>>& target) const {
        const auto &target_vocabulary = _model->get_target_vocabulary();
        const std::string *prefix = _model->decoder_start_token();

        return target_vocabulary.to_ids(target, 0, prefix, nullptr);
      }

      std::vector<std::vector<size_t>>
      WhisperNmtReplica::make_suffix_ids(const std::vector<std::vector<std::string>>& target) const {
        const auto &target_vocabulary = _model->get_target_vocabulary();
        const std::string *suffix = &target_vocabulary.eos_token();
        std::vector<std::vector<std::string>> ids;
        ids.resize(target.size());

        return target_vocabulary.to_ids(ids , 0, nullptr, suffix);
      }

      std::vector<WhisperNmtTranslationResult>
      WhisperNmtReplica::_run_translation(StorageView &features,
                                          const std::vector<std::vector<std::string>> &source_prefix,
                                          const std::vector<std::vector<std::string>> &target_prefix,
                                          const WhisperNmtOptions &options) {
        const auto scoped_device_setter = _model->get_scoped_device_setter();
        const auto device = _model->device();
        PROFILE("EncoderDecoderReplica::run_translation");

        const size_t batch_size = features.shape()[0];

        const auto sources_prefix = make_source_prefix_ids(source_prefix);
        const auto start_tokens = make_target_prefix_ids(target_prefix);

        const auto sources_suffix = make_suffix_ids(target_prefix);
        // Encode the sequence.
        StorageView memory(_encoder->output_type(), device);
        StorageView memory_lengths(DataType::INT32, device);
        _nmt_encode(features, sources_prefix, sources_suffix, memory, memory_lengths);

        layers::DecoderState state = _transformer_decoder->initial_state();
        state.emplace("memory", std::move(memory));
        state.emplace("memory_lengths", std::move(memory_lengths));

        const auto &target_vocabulary = _model->get_target_vocabulary();
        std::vector<size_t> restrict_ids;
        _transformer_decoder->update_output_layer(_model->preferred_size_multiple(), restrict_ids);

        // Decode.
        DecodingOptions decoding_options;
        decoding_options.beam_size = options.beam_size;
        decoding_options.patience = options.patience;
        decoding_options.length_penalty = options.length_penalty;
        decoding_options.coverage_penalty = options.coverage_penalty;
        decoding_options.repetition_penalty = options.repetition_penalty;
        decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
        decoding_options.prefix_bias_beta = options.prefix_bias_beta;
        decoding_options.max_length = options.max_decoding_length;
        decoding_options.min_length = options.min_decoding_length;
        decoding_options.sampling_topk = options.sampling_topk;
        decoding_options.sampling_topp = options.sampling_topp;
        decoding_options.sampling_temperature = options.sampling_temperature;
        decoding_options.num_hypotheses = options.num_hypotheses;
        decoding_options.return_scores = options.return_scores;
        decoding_options.return_logits_vocab = options.return_logits_vocab;
        decoding_options.return_attention = options.return_attention || options.replace_unknowns;
        decoding_options.return_alternatives = options.return_alternatives;
        decoding_options.min_alternative_expansion_prob = options.min_alternative_expansion_prob;
        decoding_options.disable_sequences = target_vocabulary.to_ids(options.suppress_sequences,
                /*max_length=*/0,
                /*prefix=*/nullptr,
                /*suffix=*/nullptr,
                /*allow_unk=*/false);

        if (options.disable_unk)
          decoding_options.disable_ids.push_back(target_vocabulary.unk_id());

        const auto end_ids(std::visit(ResolveEndToken(target_vocabulary), options.end_token));
        std::vector<DecodingResult> results = decode(*_transformer_decoder,
                                                     state,
                                                     start_tokens,
                                                     end_ids,
                                                     decoding_options);

        // Convert generated ids to tokens.
        std::vector<WhisperNmtTranslationResult> final_results;
        final_results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
          DecodingResult &result = results[i];

          // Remove EOS token.
          if (!options.return_end_token) {
            for (size_t h = 0; h < result.hypotheses.size(); ++h) {
              while (!result.hypotheses[h].empty() && is_eos(result.hypotheses[h].back(), end_ids)) {
                result.hypotheses[h].pop_back();
                if (!result.attention.empty())
                  result.attention[h].pop_back();
              }
            }
          }

          auto hypotheses = target_vocabulary.to_tokens(result.hypotheses);

          WhisperNmtTranslationResult final_result;
          final_result.sequences = std::move(hypotheses);
          //final_result.sequences_ids = std::move(result.hypotheses);
          final_result.scores = std::move(result.scores);
          final_result.attention = std::move(result.attention);
          final_result.logits = std::move(result.logits_vocab);
          //if (options.return_no_speech_prob)
          //  final_result.no_speech_prob = no_speech_probs[i];

          final_results.emplace_back(std::move(final_result));
        }

        return final_results;
      }

      static void remove_padding(StorageView &x, dim_t axis, dim_t size) {
        const dim_t max_size = x.dim(axis);

        if (size < max_size) {
          StorageView content(x.dtype(), x.device());
          StorageView padding(x.dtype(), x.device());

          const ops::Split split_op(axis, {size, max_size - size});
          split_op(x, content, padding);

          x = std::move(content);
        }
      }

      static std::vector<std::vector<std::pair<dim_t, dim_t>>>
      compute_alignments(StorageView &attention_probs,
                         const std::vector<size_t> &start_sequence,
                         const std::vector<std::vector<size_t>> &text_tokens,
                         const dim_t median_filter_width) {
        const ops::MedianFilter median_filter_op(median_filter_width);
        const dim_t batch_size = attention_probs.dim(0);

        // The remaining operations are not implemented on GPU, so move back to CPU.
        attention_probs.move_to(Device::CPU, DataType::FLOAT32);

        ops::LayerNorm(-2, 0)(attention_probs);

        StorageView median_filter;
        median_filter_op(attention_probs, median_filter);

        StorageView weights;
        ops::Mean(1)(median_filter, weights);

        std::vector<std::vector<std::pair<dim_t, dim_t>>> alignments;
        alignments.reserve(batch_size);

        for (dim_t b = 0; b < batch_size; ++b) {
          const dim_t text_length = text_tokens[b].size();
          const dim_t sot_length = start_sequence.size();

          StorageView matrix(Shape{text_length + 1, weights.dim(2)});
          if (weights)
            matrix.view(weights.index<float>({b, sot_length, 0}), matrix.shape());

          alignments.emplace_back(negative_dtw(matrix));
        }

        return alignments;
      }

      bool WhisperNmt::is_multilingual() const {
        const auto &replica = get_first_replica();
        return replica.is_multilingual();
      }

      size_t WhisperNmt::n_mels() const {
        const auto &replica = get_first_replica();
        return replica.n_mels();
      }

      size_t WhisperNmt::num_languages() const {
        const auto &replica = get_first_replica();
        return replica.num_languages();
      }

      std::future<StorageView> WhisperNmt::encode(const StorageView &features, const bool to_cpu) {
        return post<StorageView>(
                [features = features.sync_copy(), to_cpu](WhisperNmtReplica &replica) mutable {
                    return replica.encode(std::move(features), to_cpu);
                });
      }

      std::vector<std::future<WhisperNmtTranslationResult>>
      WhisperNmt::translate(const StorageView &features,
                            const std::vector<std::vector<std::string>>& source_prefix,
                            const std::vector<std::vector<std::string>>& target_prefix,
                            WhisperNmtOptions options) {
        const size_t batch_size = features.dim(0);
        return post_batch<WhisperNmtTranslationResult>(
                [features = features.sync_copy(),
                        source_prefix = std::move(source_prefix),
                        target_prefix = std::move(target_prefix),
                        options = std::move(options)]
                        (WhisperNmtReplica &replica) mutable {
                    return replica.translate(std::move(features), source_prefix, target_prefix, options);
                },
                batch_size);
      }
  }
}

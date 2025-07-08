#include "module.h"

#include <ctranslate2/models/whisper_nmt.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    using BatchTokensOptional = std::optional<std::vector<std::optional<Tokens>>>;

    static BatchTokens finalize_optional_batch(const BatchTokensOptional& optional) {
      // Convert missing values to empty vectors.
      BatchTokens batch;
      if (!optional)
        return batch;
      batch.reserve(optional->size());
      for (const auto& tokens : *optional) {
        batch.emplace_back(tokens.value_or(Tokens()));
      }
      return batch;
    }


    class WhisperNmtWrapper : public ReplicaPoolHelper<models::WhisperNmt> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      bool is_multilingual() const {
        return _pool->is_multilingual();
      }

      size_t n_mels() const {
        return _pool->n_mels();
      }

      size_t num_languages() const {
        return _pool->num_languages();
      }

      StorageView encode(const StorageView &features, const bool to_cpu) {
        return _pool->encode(features, to_cpu).get();
      }

      std::variant<std::vector<models::WhisperNmtTranslationResult>,
        std::vector<AsyncResult<models::WhisperNmtTranslationResult>>>
      translate(const StorageView &features,
               const BatchTokensOptional& source_prefix,
               const BatchTokensOptional& target_prefix,
               bool asynchronous,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float coverage_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               bool return_attention,
               bool return_logits_vocab,
               bool return_no_speech_prob,
               bool return_alternatives,
               float min_alternative_expansion_prob,
               size_t max_initial_timestamp_index,
               bool suppress_blank,
               const std::optional<std::vector<int>> &suppress_tokens,
               size_t sampling_topk,
               float sampling_topp,
               float sampling_temperature,
               bool replace_unknowns,
               float prefix_bias_beta,
               size_t max_decoding_length,
               size_t min_decoding_length,
               bool disable_unk,
               const std::optional<EndToken> &end_token,
               bool return_end_token,
               size_t max_input_length
      ) {
        std::vector<std::future<models::WhisperNmtTranslationResult>> futures;

        models::WhisperNmtOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.coverage_penalty = coverage_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_topp = sampling_topp;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;
        options.return_attention = return_attention;
        options.return_logits_vocab = return_logits_vocab;
        options.return_no_speech_prob = return_no_speech_prob;
        options.return_alternatives = return_alternatives;
        options.min_alternative_expansion_prob = min_alternative_expansion_prob;
        options.max_initial_timestamp_index = max_initial_timestamp_index;
        options.suppress_blank = suppress_blank;
        options.replace_unknowns = replace_unknowns;
        options.prefix_bias_beta = prefix_bias_beta;
        options.max_decoding_length = max_decoding_length;
        options.min_decoding_length = min_decoding_length;
        options.disable_unk = disable_unk;
        options.return_end_token = return_end_token;
        options.max_input_length = max_input_length;
        if (end_token)
          options.end_token = end_token.value();

        if (suppress_tokens)
          options.suppress_tokens = suppress_tokens.value();
        else
          options.suppress_tokens.clear();
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        futures = _pool->translate(features,
                                   finalize_optional_batch(source_prefix),
                                   finalize_optional_batch(target_prefix),
                                   options);

        return maybe_wait_on_futures(std::move(futures), asynchronous);
      }
    };

    void register_whisper_nmt(py::module& m) {
      py::class_<models::WhisperNmtTranslationResult>(m, "WhisperNmtGenerationResult",
                                                  "A generation result from the Whisper model.")

        .def_readonly("sequences", &models::WhisperNmtTranslationResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("scores", &models::WhisperNmtTranslationResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")
        .def_readonly("logits", &models::WhisperNmtTranslationResult::attention,
                      "logits in each sequence (empty if :obj:`return_logits_vocab` was disabled).")
        .def_readonly("no_speech_prob", &models::WhisperNmtTranslationResult::no_speech_prob,
                      "Probability of the no speech token (0 if :obj:`return_no_speech_prob` was disabled).")

        .def("__repr__", [](const models::WhisperNmtTranslationResult& result) {
          return "WhisperNmtGenerationResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            //+ ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ", logits=" + std::string(py::repr(py::cast(result.logits)))
            + ", no_speech_prob=" + std::string(py::repr(py::cast(result.no_speech_prob)))
            + ")";
        })
        ;

      declare_async_wrapper<models::WhisperNmtTranslationResult>(m, "WhisperNmtGenerationResultAsync");

      py::class_<WhisperNmtWrapper>(
        m, "WhisperNmt",
        R"pbdoc(
            Implements the Whisper speech recognition model published by OpenAI.

            See Also:
               https://github.com/openai/whisper
        )pbdoc")

        .def_property_readonly("is_multilingual", &WhisperNmtWrapper::is_multilingual,
                               "Returns ``True`` if this model is multilingual.")

        .def_property_readonly("n_mels", &WhisperNmtWrapper::n_mels,
                               "Returns dimension of mel input features.")

        .def_property_readonly("num_languages", &WhisperNmtWrapper::num_languages,
                               "Returns the number of languages supported.")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Whisper model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8, int8_float32,
                     int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                   max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                     0 for an automatic value). When the queue is full, future requests will block
                     until a free slot is available.
                   flash_attention: run model with flash attention 2 for self-attention layer
                   tensor_parallel: run model with tensor parallel mode
                   files: Load model files from the memory. This argument is a dictionary mapping
                     file names to file contents as file-like or bytes objects. If this is set,
                     :obj:`model_path` acts as an identifier for this model.
             )pbdoc")

        .def_property_readonly("device", &WhisperNmtWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &WhisperNmtWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &WhisperNmtWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &WhisperNmtWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &WhisperNmtWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &WhisperNmtWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &WhisperNmtWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        /*.def("encode", &WhisperNmtWrapper::encode,
             py::arg("features"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``.
                   to_cpu: Copy the encoder output to the CPU before returning the value.

                 Returns:
                   The encoder output.
             )pbdoc")*/

        .def("translate", &WhisperNmtWrapper::translate,
             py::arg("features"),
             py::arg("source_prefix")=py::none(),
             py::arg("target_prefix")=py::none(),
             py::kw_only(),
             py::arg("asynchronous")=false,
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("coverage_penalty")=0,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("return_attention")=false,
             py::arg("return_logits_vocab")=false,
             py::arg("return_no_speech_prob")=false,
             py::arg("return_alternatives")=false,
             py::arg("min_alternative_expansion_prob")=0,
             py::arg("max_initial_timestamp_index")=50,
             py::arg("suppress_blank")=true,
             py::arg("suppress_tokens")=std::vector<int>{-1},
             py::arg("sampling_topk")=1,
             py::arg("sampling_topp")=1,
             py::arg("sampling_temperature")=1,
             py::arg("replace_unknowns")=false,
             py::arg("prefix_bias_beta")=0,
             py::arg("max_decoding_length")=256,
             py::arg("min_decoding_length")=1,
             py::arg("disable_unk")=false,
             py::arg("end_token")=py::none(),
             py::arg("return_end_token")=false,
             py::arg("max_input_length")=1024,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input features and translate from the given audio.

                 Arguments:
                   features: Mel spectogram of the audio, as a float array with shape
                     ``[batch_size, n_mels, chunk_length]``. This method also accepts the encoded
                     features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                     which have shape ``[batch_size, chunk_length // 2, d_model]``.
                   prompts: Batch of initial string tokens or token IDs.
                   asynchronous: Run the model asynchronously.
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor, as described in
                     https://arxiv.org/abs/2204.05424. The decoding will continue until
                     beam_size*patience hypotheses are finished.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to the length during beam search.
                   coverage_penalty: Coverage penalty weight applied during beam search.
                   repetition_penalty: Penalty applied to the score of previously generated tokens
                     (set > 1 to penalize).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                     (set 0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include the scores in the output.
                   return_attention: Include the attention weights in the output.
                   return_logits_vocab: Include the log probs in the output
                   return_no_speech_prob: Include the probability of the no speech token in the
                     result.
                   return_alternatives: Include alternatives in the output.
                   min_alternative_expansion_prob: Minimum probability to expand an alternative.
                   max_initial_timestamp_index: Maximum index of the first predicted timestamp.
                   suppress_blank: Suppress blank outputs at the beginning of the sampling.
                   suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                     of symbols as defined in the model ``config.json`` file.
                   sampling_topk: Randomly sample predictions from the top K candidates.
                   sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
                   sampling_temperature: Sampling temperature to generate more random samples.
                   replace_unknowns: Replace unknown target tokens by the source token with the highest attention.
                   prefix_bias_beta: Parameter for biasing translations towards given prefix.
                   max_decoding_length: Maximum prediction length.
                   min_decoding_length: Minimum prediction length.
                   disable_unk: Disable the generation of the unknown token.
                   end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
                   return_end_token: Include the end token in the results.
                   max_input_length: Truncate inputs after this many tokens (set 0 to disable).

                 Returns:
                   A list of generation results.
             )pbdoc")

        .def("unload_model", &WhisperNmtWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model attached to this whisper but keep enough runtime context
                 to quickly resume whisper on the initial device.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
             )pbdoc")

        .def("load_model", &WhisperNmtWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the model cache in the CPU memory is not deleted if it exists.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &WhisperNmtWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready to be used.")
        ;
    }

  }
}

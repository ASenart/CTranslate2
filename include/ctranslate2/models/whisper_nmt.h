#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperNmtOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 5;

      // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
      // The decoding will continue until beam_size*patience hypotheses are finished.
      float patience = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Coverage penalty weight applied during beam search.
      float coverage_penalty = 0;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // Keep the most probable tokens whose cumulative probability exceeds this value.
      float sampling_topp = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Store attention vectors in the TranslationResult class.
      bool return_attention = false;

      // Include log probs of each token in the result
      bool return_logits_vocab = false;

      // Include the probability of the no speech token in the result.
      bool return_no_speech_prob = false;

      // Maximum index of the first predicted timestamp.
      size_t max_initial_timestamp_index = 50;

      // Suppress blank outputs at the beginning of the sampling.
      bool suppress_blank = true;

      // Return alternatives at the first unconstrained decoding position. This is typically
      // used with a target prefix to provide alternatives at a specifc location in the
      // translation.
      bool return_alternatives = false;
      // Minimum probability to expand an alternative.
      float min_alternative_expansion_prob = 0;

      // List of token IDs to suppress.
      // -1 will suppress a default set of symbols as defined in the model config.json file.
      std::vector<int> suppress_tokens = {-1};

      // Replace unknown target tokens by the original source token with the highest attention.
      bool replace_unknowns = false;

      // Biases decoding towards a given prefix, see https://arxiv.org/abs/1912.03393 --section 4.2
      // Only activates biased-decoding when beta is in range (0, 1) and SearchStrategy is set to BeamSearch.
      // The closer beta is to 1, the stronger the bias is towards the given prefix.
      //
      // If beta <= 0 and a non-empty prefix is given, then the prefix will be used as a
      // hard-prefix rather than a soft, biased-prefix.
      float prefix_bias_beta = 0;

      // Decoding length constraints.
      size_t max_decoding_length = 256;
      size_t min_decoding_length = 1;

      // Disable the generation of some sequences of tokens.
      std::vector<std::vector<std::string>> suppress_sequences;

      // Disable the generation of the unknown token.
      bool disable_unk = false;

      // Stop the decoding on one of these tokens (defaults to the model EOS token).
      std::variant<std::string, std::vector<std::string>, std::vector<size_t>> end_token;

      // Include the end token in the result.
      bool return_end_token = false;

      // Truncate the inputs after this many tokens (set 0 to disable truncation).
      size_t max_input_length = 1024;
    };

    struct WhisperNmtTranslationResult {
      std::vector<std::vector<std::string>> sequences;
      //std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;
      std::vector<std::vector<std::vector<float>>> attention;
      std::vector<std::vector<StorageView>> logits;
      float no_speech_prob = 0;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    struct WhisperNmtAlignmentResult {
      std::vector<std::pair<dim_t, dim_t>> alignments;
      std::vector<float> text_token_probs;
    };

    class WhisperNmtModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;
      size_t num_source_vocabularies() const;
      const Vocabulary& get_source_vocabulary(size_t index = 0) const;
      const Vocabulary& get_target_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

      bool with_source_bos() const {
        return config["add_source_bos"];
      }

      bool with_source_eos() const {
        return config["add_source_eos"];
      }

      const std::string* decoder_start_token() const {
        auto& start_token = config["decoder_start_token"];
        return start_token.is_null() ? nullptr : start_token.get_ptr<const std::string*>();
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
      std::vector<std::shared_ptr<const Vocabulary>> _source_vocabularies;
      std::shared_ptr<const Vocabulary> _target_vocabulary;

      void load_vocabularies(ModelReader& model_reader);
    };

    class WhisperNmtReplica : public ModelReplica {
    public:
      static std::unique_ptr<WhisperNmtReplica> create_from_model(const Model& model);

      WhisperNmtReplica(const std::shared_ptr<const WhisperNmtModel>& model);

      bool is_multilingual() const {
        return _is_multilingual;
      }

      size_t n_mels() const {
        return _n_mels;
      }

      size_t num_languages() const {
        return _num_languages;
      }

      StorageView encode(StorageView features, const bool to_cpu);

      std::vector<WhisperNmtTranslationResult>
      translate(StorageView features,
                const std::vector<std::vector<std::string>>& source_prefix,
                const std::vector<std::vector<std::string>>& target_prefix,
                const WhisperNmtOptions& options);

      std::vector<std::vector<size_t>>
      make_source_prefix_ids(const std::vector<std::vector<std::string>>& source) const;

      std::vector<std::vector<size_t>>
      make_target_prefix_ids(const std::vector<std::vector<std::string>>& target) const;

      std::vector<std::vector<size_t>>
      make_suffix_ids(const std::vector<std::vector<std::string>>& target) const;

    private:
      std::vector<WhisperNmtTranslationResult>
      _run_translation(StorageView& features,
                       const std::vector<std::vector<std::string>>& source_prefix,
                       const std::vector<std::vector<std::string>>& target_prefix,
                       const WhisperNmtOptions& options);
      void
      _nmt_encode(StorageView& features_ids,
                  const std::vector<std::vector<size_t>>& sources_prefix,
                  const std::vector<std::vector<size_t>>& sources_suffix,
                  StorageView& memory,
                  StorageView& memory_lengths);
      const std::shared_ptr<const WhisperNmtModel> _model;
      const std::unique_ptr<layers::WhisperEncoder> _encoder;
      const std::unique_ptr<layers::TransformerEncoder> _transformer_encoder;
      const std::unique_ptr<layers::TransformerDecoder> _transformer_decoder;
      const std::unique_ptr<layers::WhisperConnector> _connector;

      size_t _sot_id;
      size_t _eot_id;
      size_t _no_timestamps_id;
      size_t _no_speech_id;
      size_t _n_mels;
      size_t _num_languages;
      bool _is_multilingual;

      StorageView maybe_encode(StorageView features);
    };

    class WhisperNmt : public ReplicaPool<WhisperNmtReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      bool is_multilingual() const;
      size_t n_mels() const;
      size_t num_languages() const;

      std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

      std::vector<std::future<WhisperNmtTranslationResult>>
      translate(const StorageView& features,
                const std::vector<std::vector<std::string>>& source_prefix,
                const std::vector<std::vector<std::string>>& target_prefix,
                WhisperNmtOptions options = {});
    };

  }
}

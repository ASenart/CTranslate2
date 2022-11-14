#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"
#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {
  namespace models {

    struct WhisperOptions {
      // Beam size to use for beam search (set 1 to run greedy search).
      size_t beam_size = 1;

      // Exponential penalty applied to the length during beam search.
      float length_penalty = 1;

      // Penalty applied to the score of previously generated tokens, as described in
      // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
      float repetition_penalty = 1;

      // Prevent repetitions of ngrams with this size (set 0 to disable).
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Randomly sample from the top K candidates (set 0 to sample from the full distribution).
      size_t sampling_topk = 1;

      // High temperatures increase randomness.
      float sampling_temperature = 1;

      // Number of hypotheses to include in the result.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;
    };

    class WhisperModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class WhisperReplica : public ModelReplica {
    public:
      static std::unique_ptr<WhisperReplica> create_from_model(const Model& model);

      WhisperReplica(const std::shared_ptr<const WhisperModel>& model);

      std::vector<GenerationResult>
      generate(const StorageView& features,
               const std::vector<std::vector<std::string>>& prompts,
               const WhisperOptions& options);

      std::vector<GenerationResult>
      generate(const StorageView& features,
               const std::vector<std::vector<size_t>>& prompts,
               const WhisperOptions& options);

      std::vector<std::vector<std::pair<std::string, float>>>
      detect_language(const StorageView& features);

    private:
      const std::shared_ptr<const WhisperModel> _model;
      const std::unique_ptr<layers::WhisperEncoder> _encoder;
      const std::unique_ptr<layers::Decoder> _decoder;

      StorageView encode(const StorageView& features);
    };

    class Whisper : public ReplicaPool<WhisperReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::vector<std::future<GenerationResult>>
      generate(StorageView features,
               std::vector<std::vector<std::string>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<GenerationResult>>
      generate(StorageView features,
               std::vector<std::vector<size_t>> prompts,
               WhisperOptions options = {});

      std::vector<std::future<std::vector<std::pair<std::string, float>>>>
      detect_language(StorageView features);

    };

  }
}
import argparse
import os

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import (
    common_spec,
    transformer_spec,
    whispernmt_spec,
)

import huggingface_hub
import transformers
import itertools
from typing import List, Optional

class WhisperLoader:
    def __call__(self, model, tokenizer, spec):
        self.get_model_spec(model, spec)
        self.set_config(spec.config, model, tokenizer)
        #tokens = self.get_vocabulary(model, tokenizer)
        #self.set_vocabulary(spec, tokens)
        return spec
    @property
    def architecture_name(self):
        return "WhisperForConditionalGeneration"

    def get_model_spec(self, model, spec):
        self.set_encoder(spec.whisper_encoder, model.model.encoder)
        return spec

    def _get_lang_ids_from_tokenizer(self, tokenizer):
        non_lang_special_tokens = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nocaptions|>",
            "<|notimestamps|>",
        ]
        return [
            token_id
            for token_id, token in zip(
                tokenizer.additional_special_tokens_ids,
                tokenizer.additional_special_tokens,
            )
            if token not in non_lang_special_tokens
        ]

    def set_config(self, config, model, tokenizer):
        gen_config = getattr(model, "generation_config", None)

        if gen_config is not None:
            config.suppress_ids = gen_config.suppress_tokens
            config.suppress_ids_begin = gen_config.begin_suppress_tokens
            if hasattr(gen_config, "alignment_heads"):
                config.alignment_heads = gen_config.alignment_heads
            if hasattr(gen_config, "lang_to_id"):
                config.lang_ids = sorted(gen_config.lang_to_id.values())
        else:
            config.suppress_ids = model.config.suppress_tokens
            config.suppress_ids_begin = model.config.begin_suppress_tokens
            config.alignment_heads = _WHISPER_ALIGNMENT_HEADS.get(model.name_or_path)

        if getattr(config, "lang_ids", None) is None:
            config.lang_ids = self._get_lang_ids_from_tokenizer(tokenizer)

        if config.alignment_heads is None:
            # Use the last half layers for alignment by default.
            num_layers = model.config.decoder_layers
            num_heads = model.config.decoder_attention_heads
            config.alignment_heads = list(
                itertools.product(
                    range(num_layers // 2, num_layers),
                    range(num_heads),
                )
            )

    def set_linear(self, spec, module):
        spec.weight = module.weight

        if isinstance(module, transformers.Conv1D):
            spec.weight = spec.weight.transpose(0, 1)
        if module.bias is not None:
            spec.bias = module.bias

    def set_attention(self, spec, attention, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q_proj)
        self.set_linear(split_layers[1], attention.k_proj)
        self.set_linear(split_layers[2], attention.v_proj)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.out_proj)

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "offset", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset:]

    def set_embeddings(self, spec, module):
        spec.weight = module.weight

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight
        spec.beta = module.bias

    def get_vocabulary(self, model, tokenizer):
        tokens = [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]

        # Add timestamp tokens.
        tokens.extend(
            "<|%.2f|>" % (i * 0.02)
            for i in range(model.config.vocab_size - len(tokens))
        )

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_encoder(self, spec, encoder):
        self.set_conv1d(spec.conv1, encoder.conv1)
        self.set_conv1d(spec.conv2, encoder.conv2)
        self.set_common_layers(spec, encoder)

        for layer_spec, layer in zip(spec.layer, encoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_decoder(self, spec, decoder):
        self.set_embeddings(spec.embeddings, decoder.embed_tokens)
        self.set_common_layers(spec, decoder)

        for layer_spec, layer in zip(spec.layer, decoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            if hasattr(layer, "encoder_attn"):
                self.set_attention(
                    layer_spec.attention,
                    layer.encoder_attn,
                    self_attention=False,
                )
                self.set_layer_norm(
                    layer_spec.attention.layer_norm,
                    layer.encoder_attn_layer_norm,
                )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_common_layers(self, spec, module):
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_conv1d(self, spec, module):
        spec.weight = module.weight
        spec.bias = module.bias

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "fast_gelu": common_spec.Activation.GELUTanh,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
}

_SUPPORTED_FEATURES_MERGE = {
    "concat": common_spec.EmbeddingsMerge.CONCAT,
    "sum": common_spec.EmbeddingsMerge.ADD,
}


def check_opt(opt, num_source_embeddings):
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    feat_merge = getattr(opt, "feat_merge", "concat")
    self_attn_type = getattr(opt, "self_attn_type", "scaled-dot")

    check = utils.ConfigurationChecker()
    check(
        opt.encoder_type == opt.decoder_type
        and opt.decoder_type in {"transformer", "transformer_lm"},
        "Options --encoder_type and --decoder_type must be"
        " 'transformer' or 'transformer_lm",
    )
    check(
        self_attn_type == "scaled-dot",
        "Option --self_attn_type %s is not supported (supported values are: scaled-dot)"
        % self_attn_type,
    )
    check(
        activation_fn in _SUPPORTED_ACTIVATIONS,
        "Option --pos_ffn_activation_fn %s is not supported (supported activations are: %s)"
        % (activation_fn, ", ".join(_SUPPORTED_ACTIVATIONS.keys())),
    )
    check(
        opt.position_encoding != (with_relative_position or with_rotary or with_alibi),
        "Options --position_encoding and --max_relative_positions cannot be both enabled "
        "or both disabled",
    )
    check(
        num_source_embeddings == 1 or feat_merge in _SUPPORTED_FEATURES_MERGE,
        "Option --feat_merge %s is not supported (supported merge modes are: %s)"
        % (feat_merge, " ".join(_SUPPORTED_FEATURES_MERGE.keys())),
    )
    check.validate()


def _get_model_spec_seq2seq(
    opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings, whisper_encoder_num_layers, whisper_encoder_num_heads
):
    """Creates a model specification from the model options."""
    activation_fn = "relu"

    # Return the first head of the last layer unless the model was trained with alignments.
    if getattr(opt, "lambda_align", 0) == 0:
        alignment_layer = -1
        alignment_heads = 1
    else:
        alignment_layer = opt.alignment_layer
        alignment_heads = opt.alignment_heads

    num_heads = getattr(opt, "num_heads", 16)
    num_layers = getattr(opt, "num_layers", 6)

    model_spec = whispernmt_spec.WhisperNmtSpec.from_config(
        num_layers,
        num_heads,
        whisper_encoder_num_layers,
        whisper_encoder_num_heads,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        alignment_layer=alignment_layer,
        alignment_heads=alignment_heads,
        num_source_embeddings=num_source_embeddings,
        multi_query_attention=getattr(opt, "multiquery", False),
    )

    model_spec.config.decoder_start_token = getattr(opt, "decoder_start_token", "<s>")

    set_transformer_spec(model_spec, variables)
    for src_vocab in src_vocabs:
        model_spec.register_source_vocabulary(src_vocab)
    for tgt_vocab in tgt_vocabs:
        model_spec.register_target_vocabulary(tgt_vocab)

    return model_spec


def _get_model_spec_lm(opt, variables, src_vocabs, tgt_vocabs, num_source_embeddings):
    """Creates a model specification from the model options."""
    with_relative_position = getattr(opt, "max_relative_positions", 0) > 0
    with_rotary = getattr(opt, "max_relative_positions", 0) == -1
    with_alibi = getattr(opt, "max_relative_positions", 0) == -2
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    num_heads = getattr(opt, "heads", 8)
    num_kv = getattr(opt, "num_kv", 0)
    if num_kv == num_heads or num_kv == 0:
        num_kv = None
    rotary_dim = 0 if with_rotary else None
    rotary_interleave = getattr(opt, "rotary_interleave", True)
    ffn_glu = activation_fn == "silu"
    sliding_window = getattr(opt, "sliding_window", 0)

    model_spec = transformer_spec.TransformerDecoderModelSpec.from_config(
        opt.dec_layers,
        num_heads,
        activation=_SUPPORTED_ACTIVATIONS[activation_fn],
        ffn_glu=ffn_glu,
        with_relative_position=with_relative_position,
        alibi=with_alibi,
        rms_norm=opt.layer_norm == "rms",
        rotary_dim=rotary_dim,
        rotary_interleave=rotary_interleave,
        multi_query_attention=getattr(opt, "multiquery", False),
        num_heads_kv=num_kv,
        sliding_window=sliding_window,
    )

    model_spec.config.layer_norm_epsilon = getattr(opt, "norm_eps", 1e-6)

    set_transformer_decoder(
        model_spec.decoder,
        variables,
        with_encoder_attention=False,
    )

    for tgt_vocab in tgt_vocabs:
        model_spec.register_vocabulary(tgt_vocab)

    return model_spec


def get_vocabs(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    src_vocabs = [vocab]
    tgt_vocabs = [vocab]

    return src_vocabs, tgt_vocabs


class WhisperNMTConverter(Converter):
    """Converts models generated by OpenNMT-py."""

    def __init__(self, model_path: str,
                 whisper_model: str,
                 vocab_path: str,
                 copy_files: Optional[List[str]] = None,
                 revision: Optional[str] = None,
                 low_cpu_mem_usage: bool = False,
                 trust_remote_code: bool = False):
        """Initializes the OpenNMT-py converter.

        Arguments:
          model_path: Path to the OpenNMT-py PyTorch model (.pt file).
        """
        self._model_path = model_path
        self._vocab_path = vocab_path
        self._whisper_model = whisper_model
        self._copy_files = copy_files
        self._revision = revision
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._trust_remote_code = trust_remote_code

    def _load(self):
        import torch
        with torch.no_grad():
            config = transformers.AutoConfig.from_pretrained(
                self._whisper_model, trust_remote_code=self._trust_remote_code
            )
            loader = WhisperLoader()
            model_class = getattr(transformers, loader.architecture_name)
            tokenizer_class = transformers.AutoTokenizer

            kwargs = {
                "torch_dtype": (
                    getattr(config, "torch_dtype", None)
                )
            }

            if self._revision:
                kwargs["revision"] = self._revision
            if self._low_cpu_mem_usage:
                kwargs["low_cpu_mem_usage"] = self._low_cpu_mem_usage
            if self._trust_remote_code:
                kwargs["trust_remote_code"] = self._trust_remote_code

            model = self.load_model(model_class, self._whisper_model, **kwargs)
            whisper_encoder_num_layers = model.config.encoder_layers
            whisper_encoder_num_heads = model.config.encoder_attention_heads
            tokenizer_kwargs = {}
            if self._trust_remote_code:
                tokenizer_kwargs["trust_remote_code"] = self._trust_remote_code

            tokenizer = self.load_tokenizer(
                tokenizer_class, self._whisper_model, **tokenizer_kwargs
            )

            checkpoint = torch.load(self._model_path, map_location="cpu")
            src_vocabs, tgt_vocabs = get_vocabs(self._vocab_path)

            variables = checkpoint["model"]
            spec = _get_model_spec_seq2seq(
                checkpoint["model_config"],
                variables,
                src_vocabs,
                tgt_vocabs,
                num_source_embeddings=len(src_vocabs),
                whisper_encoder_num_layers=whisper_encoder_num_layers,
                whisper_encoder_num_heads=whisper_encoder_num_heads
            )

            spec = loader(model, tokenizer, spec)
            if self._copy_files:
                for filename in self._copy_files:
                    spec.register_file(self.get_model_file(filename))
        return spec

    def load_model(self, model_class, model_name_or_path, **kwargs):
        return model_class.from_pretrained(model_name_or_path, **kwargs)

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        return tokenizer_class.from_pretrained(model_name_or_path, **kwargs)

    def get_model_file(self, filename):
        if os.path.isdir(self._whisper_model):
            path = os.path.join(self._whisper_model, filename)
        else:
            try:
                path = huggingface_hub.hf_hub_download(
                    repo_id=self._whisper_model, filename=filename
                )
            except huggingface_hub.utils.EntryNotFoundError:
                path = None

        if path is None or not os.path.isfile(path):
            raise ValueError(
                "File %s does not exist in model %s"
                % (filename, self._whisper_model)
            )

        return path


def set_transformer_spec(spec, variables):
    #set_whisper_encoder(spec.whisper, variables)
    set_whisper_reshape(spec.connector, variables)
    set_transformer_encoder(spec.transformer_encoder, variables)
    set_transformer_decoder(spec.transformer_decoder, variables)

def set_whisper_reshape(spec, variables):
    set_linear(spec.linear1, variables, "lin_1")
    set_linear(spec.linear2, variables, "lin_2")

def set_transformer_encoder(spec, variables):
    set_input_layers(spec, variables, "src_embeddings")
    set_layer_norm(spec.layer_norm, variables, "encoder.norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer(layer, variables, "encoder.layers.%d" % i)


def set_transformer_decoder(spec, variables, with_encoder_attention=True):
    set_input_layers(spec, variables, "tgt_embeddings")
    set_layer_norm(spec.layer_norm, variables, "decoder.norm")
    for i, layer in enumerate(spec.layer):
        set_transformer_decoder_layer(
            layer,
            variables,
            "decoder.layers.%d" % i,
            with_encoder_attention=with_encoder_attention,
        )

    set_linear(spec.projection, variables, "output_layer")
    #try:
    #    set_linear(spec.projection, variables, "generator")
    #except KeyError:
    #    # Compatibility when the generator was a nn.Sequential module.
    #    set_linear(spec.projection, variables, "generator.0")


def set_input_layers(spec, variables, scope):
    if hasattr(spec, "position_encodings"):
        set_position_encodings(
            spec.position_encodings,
            variables,
            "position_encodings",
        )
    else:
        # See https://github.com/OpenNMT/OpenNMT-py/issues/1722
        spec.scale_embeddings = False

    set_embeddings(
        (
            spec.embeddings[0]
            if isinstance(spec.embeddings, list)
            else spec.embeddings
        ),
        variables, scope
    )


def set_transformer_encoder_layer(spec, variables, scope):
    set_layer_norm(spec.ffn.layer_norm, variables, "%s.norm2" % scope)
    set_ffn(spec.ffn, variables, "%s.ffn" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attention" % scope,
        self_attention=True,
    )
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.norm1" % scope)


def set_transformer_decoder_layer(spec, variables, scope, with_encoder_attention=True):
    set_layer_norm(spec.ffn.layer_norm, variables, "%s.norm3" % scope)
    set_ffn(spec.ffn, variables, "%s.ffn" % scope)
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.self_attention" % scope,
        self_attention=True,
    )
    set_layer_norm(spec.self_attention.layer_norm, variables, "%s.norm1" % scope)
    if with_encoder_attention:
        set_multi_head_attention(spec.attention, variables, "%s.attention" % scope)
        set_layer_norm(spec.attention.layer_norm, variables, "%s.norm2" % scope)


def set_ffn(spec, variables, scope):
    set_linear(spec.linear_0, variables, "%s.inner" % scope)
    set_linear(spec.linear_1, variables, "%s.outer" % scope)
    if hasattr(spec, "linear_0_noact"):
        set_linear(spec.linear_0_noact, variables, "%s.w_3" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False):
    if self_attention:
        set_linear(spec.linear[0], variables, "%s.in_proj" % scope)
    else:
        set_linear(spec.linear[0], variables, "%s.query_proj" % scope)
        set_linear(spec.linear[1], variables, "%s.value_proj" % scope)
    set_linear(spec.linear[-1], variables, "%s.out_proj" % scope)
    if hasattr(spec, "relative_position_keys"):
        spec.relative_position_keys = _get_variable(
            variables, "%s.relative_positions_embeddings.weight" % scope
        )
        spec.relative_position_values = spec.relative_position_keys


def set_layer_norm(spec, variables, scope):
    try:
        spec.gamma = _get_variable(variables, "%s.weight" % scope)
    except KeyError:
        # Compatibility with older models using a custom LayerNorm module.
        spec.gamma = _get_variable(variables, "%s.a_2" % scope)
        spec.beta = _get_variable(variables, "%s.b_2" % scope)
    try:
        spec.beta = _get_variable(variables, "%s.bias" % scope)
    except KeyError:
        pass


def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    bias = variables.get("%s.bias" % scope)
    if bias is not None:
        spec.bias = bias


def set_embeddings(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)


def set_position_encodings(spec, variables, scope):
    spec.encodings = _get_variable(variables, "%s" % scope).squeeze()


def _get_variable(variables, name):
    return variables[name]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", required=True, help="Model path.")
    parser.add_argument("--vocab_path", required=True, help="Vocab path.")
    parser.add_argument("--whisper_model", required=True, help="Whisper model name.")
    Converter.declare_arguments(parser)
    args = parser.parse_args()
    WhisperNMTConverter(args.model_path, args.whisper_model, args.vocab_path).convert_from_args(args)


if __name__ == "__main__":
    main()

# Cross-attention heads that are highly correlated to the word-level timing,
# i.e. the alignment between audio and text tokens.
# Obtained from https://github.com/openai/whisper/blob/v20231106/whisper/__init__.py#L32-L47
_WHISPER_ALIGNMENT_HEADS = {
    "openai/whisper-tiny.en": [
        (1, 0),
        (2, 0),
        (2, 5),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
    ],
    "openai/whisper-tiny": [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
    "openai/whisper-base.en": [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
    "openai/whisper-base": [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ],
    "openai/whisper-small.en": [
        (6, 6),
        (7, 0),
        (7, 3),
        (7, 8),
        (8, 2),
        (8, 5),
        (8, 7),
        (9, 0),
        (9, 4),
        (9, 8),
        (9, 10),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 6),
        (10, 11),
        (11, 2),
        (11, 4),
    ],
    "openai/whisper-small": [
        (5, 3),
        (5, 9),
        (8, 0),
        (8, 4),
        (8, 7),
        (8, 8),
        (9, 0),
        (9, 7),
        (9, 9),
        (10, 5),
    ],
    "openai/whisper-medium.en": [
        (11, 4),
        (14, 1),
        (14, 12),
        (14, 14),
        (15, 4),
        (16, 0),
        (16, 4),
        (16, 9),
        (17, 12),
        (17, 14),
        (18, 7),
        (18, 10),
        (18, 15),
        (20, 0),
        (20, 3),
        (20, 9),
        (20, 14),
        (21, 12),
    ],
    "openai/whisper-medium": [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    "openai/whisper-large": [
        (9, 19),
        (11, 2),
        (11, 4),
        (11, 17),
        (22, 7),
        (22, 11),
        (22, 17),
        (23, 2),
        (23, 15),
    ],
    "openai/whisper-large-v2": [
        (10, 12),
        (13, 17),
        (16, 11),
        (16, 12),
        (16, 13),
        (17, 15),
        (17, 16),
        (18, 4),
        (18, 11),
        (18, 19),
        (19, 11),
        (21, 2),
        (21, 3),
        (22, 3),
        (22, 9),
        (22, 12),
        (23, 5),
        (23, 7),
        (23, 13),
        (25, 5),
        (26, 1),
        (26, 12),
        (27, 15),
    ],
    "openai/whisper-large-v3": [
        (7, 0),
        (10, 17),
        (12, 18),
        (13, 12),
        (16, 1),
        (17, 14),
        (19, 11),
        (21, 4),
        (24, 1),
        (25, 6),
    ],
}

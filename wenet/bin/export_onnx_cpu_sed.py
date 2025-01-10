# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import copy
import sys

import torch
import yaml
import numpy as np

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model

try:
    import onnx
    import onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Please install onnx and onnxruntime!")
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="export your script model")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--output_dir", required=True, help="output directory")
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def print_input_output_info(onnx_model, name, prefix="\t\t"):
    input_names = [node.name for node in onnx_model.graph.input]
    input_shapes = [
        [d.dim_value for d in node.type.tensor_type.shape.dim]
        for node in onnx_model.graph.input
    ]
    output_names = [node.name for node in onnx_model.graph.output]
    output_shapes = [
        [d.dim_value for d in node.type.tensor_type.shape.dim]
        for node in onnx_model.graph.output
    ]
    print("{}{} inputs : {}".format(prefix, name, input_names))
    print("{}{} input shapes : {}".format(prefix, name, input_shapes))
    print("{}{} outputs: {}".format(prefix, name, output_names))
    print("{}{} output shapes : {}".format(prefix, name, output_shapes))


def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    encoder = asr_model
    encoder.forward = asr_model.decode
    encoder_outpath = os.path.join(args["output_dir"], "sed.onnx")

    print("\tStage-1.1: prepare inputs for encoder")
    speech = torch.randn((args["batch"], args["decoding_window"], args["feature_size"]))
    speech_lengths = torch.IntTensor([args["decoding_window"]] * args["batch"])
    inputs = (speech, speech_lengths)
    print(
        "\t\tspeech.size(): {}\n".format(speech.size()),
        "\t\tspeech_lengths.size(): {}\n".format(speech_lengths.size()),
    )

    print("\tStage-1.2: torch.onnx.export")
    dynamic_axes = {
        "speech": {1: "T"},
    }
    torch.onnx.export(
        encoder,
        inputs,
        encoder_outpath,
        opset_version=13,
        export_params=True,
        do_constant_folding=True,
        input_names=[
            "speech",
            "speech_lengths",
        ],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    onnx_encoder = onnx.load(encoder_outpath)
    for k, v in args.items():
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    onnx.save(onnx_encoder, encoder_outpath)
    print_input_output_info(onnx_encoder, "onnx_encoder")
    # Dynamic quantization
    model_fp32 = encoder_outpath
    model_quant = os.path.join(args["output_dir"], "sed.quant.onnx")
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
    print("\t\tExport onnx_encoder, done! see {}".format(encoder_outpath))

    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_speech = copy.deepcopy(speech)
    torch_speech_lengths = copy.deepcopy(speech_lengths)
    for i in range(10):
        print(
            "\t\ttorch speech-{}: {}, speech_lengths: {}".format(
                i,
                list(torch_speech.size()),
                list(torch_speech_lengths.size()),
            )
        )
        out = encoder(
            torch_speech,
            torch_speech_lengths,
        )
        torch_output.append(out)
    torch_output = torch.cat(torch_output, dim=1)

    onnx_output = []
    onnx_speech = to_numpy(speech)
    onnx_speech_lengths = to_numpy(speech_lengths)
    ort_session = onnxruntime.InferenceSession(
        encoder_outpath, providers=["CPUExecutionProvider"]
    )
    input_names = [node.name for node in onnx_encoder.graph.input]
    for i in range(10):
        print(
            "\t\tonnx  speech-{}: {}, speech_lengths: {}".format(
                i,
                onnx_speech.shape,
                onnx_speech_lengths.shape,
            )
        )
        ort_inputs = {
            "speech": onnx_speech,
            "speech_lengths": onnx_speech_lengths,
        }
        for k in list(ort_inputs):
            if k not in input_names:
                ort_inputs.pop(k)
        ort_out = ort_session.run(None, ort_inputs)
        onnx_output.append(ort_out[0])
    onnx_output = np.concatenate(onnx_output, axis=1)

    np.testing.assert_allclose(
        to_numpy(torch_output), onnx_output, rtol=1e-03, atol=1e-05
    )
    meta = ort_session.get_modelmeta()
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\t\tCheck onnx_encoder, pass!")


def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)

    arguments = {}
    arguments["output_dir"] = output_dir
    arguments["batch"] = 1
    arguments["feature_size"] = configs["input_dim"]
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    arguments["decoding_window"] = 67

    export_encoder(model, arguments)


if __name__ == "__main__":
    main()

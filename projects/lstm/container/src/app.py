import logging
from typing import Any, cast, List
from infernet_ml.utils.common_types import TensorInput

from eth_abi import decode, encode  # type: ignore
from infernet_ml.utils.model_loader import (
    HFLoadArgs,
)
from infernet_ml.utils.model_loader import ModelSource
from infernet_ml.utils.service_models import InfernetInput, JobLocation
from infernet_ml.workflows.inference.torch_inference_workflow import (
    TorchInferenceWorkflow,
    TorchInferenceInput,
)
from quart import Quart, request

# ## REPLACE
# # Note: the IrisClassificationModel needs to be imported in this file for it to exist
# # in the classpath. This is because pytorch requires the model to be in the classpath.
# # Simply downloading the weights and model from the hub is not enough.
# from iris_classification_model import IrisClassificationModel
# TODO put our model on huggingface hub
from ritual_lstm_forecaster import LSTMForecaster

log = logging.getLogger(__name__)


"""
Idea:
- The LSTMForecaster class is a wrapper around the LSTM model.
- Onchain users can
    - register a new model
    - evaluate all models
    - evaluate top n models
    - run inference (prediction) on certain model
"""


def create_app() -> Quart:
    app = Quart(__name__)
    # we are downloading the model from the hub.
    # model repo is located at: https://huggingface.co/Ritual-Net/iris-dataset
    workflow = TorchInferenceWorkflow(
        model_source=ModelSource.HUGGINGFACE_HUB,
        load_args=HFLoadArgs(
            repo_id="TODO-our-org/TODO-our-dataset", filename="TODO.torch"
        ),
    )
    workflow.setup()

    @app.route("/")
    def index() -> str:
        """
        Utility endpoint to check if the service is running.
        """
        return f"LSTM Forecaster Example Program: {LSTMForecaster.__name__}"

    @app.route("/service_output", methods=["POST"])
    async def inference() -> dict[str, Any]:
        req_data = await request.get_json()
        """
        InfernetInput has the format:
            source: (0 on-chain, 1 off-chain)
            data: dict[str, Any]
        """
        infernet_input: InfernetInput = InfernetInput(**req_data)

        match infernet_input:
            case InfernetInput(source=JobLocation.OFFCHAIN):
                web2_input = cast(dict[str, Any], infernet_input.data)
                values = cast(List[List[float]], web2_input["input"])
            case InfernetInput(source=JobLocation.ONCHAIN):
                web3_input: List[int] = decode(
                    ["uint256[]"], bytes.fromhex(cast(str, infernet_input.data))
                )[0]
                values = [[float(v) / 1e6 for v in web3_input]]
            case _:
                raise ValueError("Invalid source")

        """
        The input to the torch inference workflow needs to conform to this format:

        {
            "dtype": str,
            "values": list[Any]
        }

        For more information refer to:
        https://infernet-ml.docs.ritual.net/reference/infernet_ml/workflows/inference/torch_inference_workflow/?h=torch

        """  # noqa: E501

        ############################
        ## REPLACE
        # TODO, need to decide what input will look like (model_name, current date, etc.)
        log.info("Input values: %s", values)

        _input = TensorInput(
            dtype="float",
            shape=(1, 4),
            values=values,
        )
        ############################

        lstm_inference_input = TorchInferenceInput(input=_input)

        inference_result = workflow.inference(lstm_inference_input)

        result = inference_result.outputs

        match infernet_input:
            case InfernetInput(destination=JobLocation.OFFCHAIN):
                """
                In case of an off-chain request, the result is returned as is.
                """
                return {"result": result}
            case InfernetInput(destination=JobLocation.ONCHAIN):
                """
                In case of an on-chain request, the result is returned in the format:
                {
                    "raw_input": str,
                    "processed_input": str,
                    "raw_output": str,
                    "processed_output": str,
                    "proof": str,
                }
                refer to: https://docs.ritual.net/infernet/node/containers for more
                info.
                """
                prediction_normalized = int(result * 1e6)
                return {
                    "raw_input": "",
                    "processed_input": "",
                    "raw_output": encode(["uint256"], prediction_normalized).hex(),
                    "processed_output": "",
                    "proof": "",
                }
            case _:
                raise ValueError("Invalid destination")

    return app


if __name__ == "__main__":
    """
    Utility to run the app locally. For development purposes only.
    """
    create_app().run(port=3000)

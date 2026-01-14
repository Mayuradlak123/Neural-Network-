from fastapi import APIRouter
import torch
import torch.nn.functional as F
from pydantic import BaseModel

class ActivationRequest(BaseModel):
    value: float

activation_router = APIRouter(
    prefix="/activations",
    tags=["Activation Functions"]
)

@activation_router.post("/run")
def run_activations(request: ActivationRequest):
    x = torch.tensor([request.value], dtype=torch.float32)

    return {
        "input": request.value,

        # Yes / No probability
        "sigmoid": torch.sigmoid(x).item(),

        # Negative vs Positive
        "tanh": torch.tanh(x).item(),

        # ON / OFF
        "relu": F.relu(x).item(),

        # OFF but little ON
        "leaky_relu": F.leaky_relu(x).item(),

        # Smooth OFF
        "elu": F.elu(x).item(),

        # Smart ReLU (Transformers)
        "gelu": F.gelu(x).item(),

        # Winner selection (normally multi-class)
        "softmax": F.softmax(x, dim=0).tolist(),

        # Smooth ReLU
        "swish": (x * torch.sigmoid(x)).item()
    }

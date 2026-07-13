from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None
    nn = None


if nn is None:

    class GatedFusionClassifier:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use neural models: pip install -r requirements.txt")

    class DiseasePrototypeHead:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch to use neural models: pip install -r requirements.txt")

    class MultimodalCXRModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install torch, timm and transformers to use MultimodalCXRModel.")

else:

    class GatedFusionClassifier(nn.Module):
        """Gated multimodal head: the gate learns image-vs-text contribution per case."""

        def __init__(
            self,
            image_dim: int,
            text_dim: int,
            hidden_dim: int = 512,
            num_labels: int = 1,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.image_proj = nn.Linear(image_dim, hidden_dim)
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )

        def fuse(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            image_h = torch.tanh(self.image_proj(image_emb))
            text_h = torch.tanh(self.text_proj(text_emb))
            gate = self.gate(torch.cat([image_h, text_h], dim=-1))
            fused = gate * image_h + (1.0 - gate) * text_h
            return fused, gate

        def forward(
            self,
            image_emb: torch.Tensor,
            text_emb: torch.Tensor,
            return_gate: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            fused, gate = self.fuse(image_emb, text_emb)
            logits = self.classifier(fused)
            if return_gate:
                return logits, gate
            return logits


    class DiseasePrototypeHead(nn.Module):


        def __init__(self, hidden_dim: int, num_labels: int):
            super().__init__()
            self.prototypes = nn.Parameter(torch.randn(num_labels, hidden_dim))

        def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
            fused_norm = nn.functional.normalize(fused_emb, dim=-1)
            proto_norm = nn.functional.normalize(self.prototypes, dim=-1)
            return fused_norm @ proto_norm.T


    class MultimodalCXRModel(nn.Module):

        def __init__(
            self,
            num_labels: int,
            image_encoder_name: str = "tf_efficientnet_b0",
            text_encoder_name: str = "emilyalsentzer/Bio_ClinicalBERT",
            hidden_dim: int = 512,
            dropout: float = 0.2,
            pretrained: bool = True,
            freeze_backbones: bool = False,
        ):
            super().__init__()
            try:
                import timm
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError("Install timm and transformers to use MultimodalCXRModel.") from exc

            self.image_encoder = timm.create_model(
                image_encoder_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            image_dim = self.image_encoder.num_features

            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
            text_dim = self.text_encoder.config.hidden_size

            if freeze_backbones:
                for parameter in self.image_encoder.parameters():
                    parameter.requires_grad = False
                for parameter in self.text_encoder.parameters():
                    parameter.requires_grad = False

            self.fusion = GatedFusionClassifier(
                image_dim=image_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                dropout=dropout,
            )

        def forward(
            self,
            pixel_values: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            return_gate: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            image_emb = self.image_encoder(pixel_values)
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_emb = text_outputs.last_hidden_state[:, 0]
            return self.fusion(image_emb=image_emb, text_emb=text_emb, return_gate=return_gate)

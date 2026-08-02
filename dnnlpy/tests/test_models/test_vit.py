import pytest
import torch
from torch.testing import assert_close

from dnnlpy.models import vit


def test_vit_linear_patch_embedding_returns_patch_tokens():
    x = torch.randn(2, 3, 8, 8)

    module = vit.ViTLinearPatchEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
    )
    output = module(x)

    assert module.num_patches == 4
    assert output.shape == (2, 4, 6)


def test_vit_conv_patch_embedding_returns_patch_tokens():
    x = torch.randn(2, 3, 8, 8)

    module = vit.ViTConvPatchEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
    )
    output = module(x)

    assert module.num_patches == 4
    assert output.shape == (2, 4, 6)


def test_vit_linear_and_conv_patch_embeddings_are_equivalent():
    x = torch.randn(2, 2, 4, 4)

    linear = vit.ViTLinearPatchEmbedding(
        image_size=4,
        patch_size=2,
        in_channels=2,
        embed_dim=3,
    )
    convolution = vit.ViTConvPatchEmbedding(
        image_size=4,
        patch_size=2,
        in_channels=2,
        embed_dim=3,
    )

    with torch.no_grad():
        convolution.proj.weight.copy_(
            linear.proj.weight.reshape_as(convolution.proj.weight)
        )
        assert linear.proj.bias is not None
        assert convolution.proj.bias is not None
        convolution.proj.bias.copy_(linear.proj.bias)

    assert_close(linear(x), convolution(x))


def test_vit_patch_embedding_requires_divisible_image_size():
    with pytest.raises(AssertionError, match='divisible'):
        vit.ViTLinearPatchEmbedding(
            image_size=7,
            patch_size=4,
            in_channels=3,
            embed_dim=6,
        )


def test_patchify_splits_images_into_flattened_patches():
    x = torch.arange(2 * 1 * 4 * 4).reshape(2, 1, 4, 4)

    patches = vit.patchify(x, patch_size=2)

    assert patches.shape == (2, 4, 4)
    assert torch.equal(patches[0, 0], torch.tensor([0, 1, 4, 5]))
    assert torch.equal(patches[0, 3], torch.tensor([10, 11, 14, 15]))


def test_patchify_requires_divisible_image_size():
    x = torch.randn(2, 3, 5, 4)

    with pytest.raises(AssertionError, match='divisible'):
        vit.patchify(x, patch_size=2)


def test_vit_positional_embedding_adds_positions_and_validates_length():
    x = torch.zeros(2, 5, 6)

    module = vit.ViTPositionalEmbedding(
        num_patches=4,
        embed_dim=6,
        use_cls_token=True,
    )
    output = module(x)

    assert output.shape == x.shape
    assert_close(output, module.pos_embed.expand_as(output))

    with pytest.raises(AssertionError, match='Expected sequence length'):
        module(torch.zeros(2, 4, 6))


def test_vit_positional_embedding_interpolates_with_and_without_cls_token():
    with_cls = vit.ViTPositionalEmbedding(
        num_patches=4,
        embed_dim=6,
        use_cls_token=True,
    )
    without_cls = vit.ViTPositionalEmbedding(
        num_patches=4,
        embed_dim=6,
        use_cls_token=False,
    )

    cls_output = with_cls.interpolate((2, 2), (3, 3))
    patch_output = without_cls.interpolate((2, 2), (3, 3))

    assert cls_output.shape == (1, 10, 6)
    assert patch_output.shape == (1, 9, 6)
    assert_close(cls_output[:, :1], with_cls.pos_embed[:, :1])


def test_vit_positional_embedding_rejects_mismatched_old_grid():
    module = vit.ViTPositionalEmbedding(num_patches=4, embed_dim=6)

    with pytest.raises(AssertionError, match='Expected old grid'):
        module.interpolate((1, 3), (2, 2))


def test_vit_embedding_returns_patch_and_class_tokens():
    x = torch.randn(2, 3, 8, 8)

    module = vit.ViTEmbedding(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
        dropout=0.0,
    )

    with torch.no_grad():
        module.pos_embed.pos_embed.zero_()

    output = module(x)
    expected_patches = module.patch_embed(x)
    resized_pos_embed = module.interpolate_pos_embedding((2, 2), (3, 3))

    assert output.shape == (2, 5, 6)
    assert_close(output[:, :1], module.add_cls_token.cls_token.expand(2, -1, -1))
    assert_close(output[:, 1:], expected_patches)
    assert resized_pos_embed.shape == (1, 10, 6)


def test_vit_mlp_uses_declared_default_hidden_dimension():
    x = torch.randn(2, 5, 6)

    module = vit.ViTMLP(embed_dim=6, dropout=0.0)
    output = module(x)

    assert module.net[0].out_features == 3072
    assert output.shape == x.shape


def test_vit_encoder_layer_returns_token_sequence():
    x = torch.randn(2, 5, 6)

    module = vit.ViTEncoderLayer(
        embed_dim=6,
        num_heads=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    output = module(x)

    assert output.shape == x.shape


def test_vit_encoder_layer_residual_path_preserves_input_with_zero_weights():
    x = torch.randn(2, 5, 6)

    module = vit.ViTEncoderLayer(
        embed_dim=6,
        num_heads=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()

    assert_close(module(x), x)


def test_vit_encoder_stacks_layers_and_normalizes_output():
    x = torch.randn(2, 5, 6)

    module = vit.ViTEncoder(
        embed_dim=6,
        num_heads=2,
        num_layers=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    output = module(x)

    assert len(module.layers) == 2
    assert output.shape == x.shape


def test_vit_classification_head_uses_class_token():
    x = torch.tensor(
        [
            [[2.0, 3.0, 4.0], [10.0, 20.0, 30.0]],
            [[5.0, 7.0, 11.0], [13.0, 17.0, 19.0]],
        ]
    )

    module = vit.ViTClassificationHead(embed_dim=3, num_classes=2)

    with torch.no_grad():
        weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        module.head.weight.copy_(weight)
        if module.head.bias is not None:
            module.head.bias.zero_()

    actual = module(x)
    expected = torch.tensor([[2.0, 3.0], [5.0, 7.0]])

    assert_close(actual, expected)


def test_vit_model_returns_encoded_tokens():
    x = torch.randn(2, 3, 8, 8)

    module = vit.ViTModel(
        image_size=8,
        patch_size=4,
        in_channels=3,
        embed_dim=6,
        num_heads=2,
        num_layers=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )
    output = module(x)

    assert output.shape == (2, 5, 6)


def test_vit_for_image_classification_returns_class_logits():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)

    module = vit.ViTForImageClassification(
        image_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=7,
        embed_dim=6,
        num_heads=2,
        num_layers=2,
        hidden_dim=12,
        dropout=0.0,
        attn_dropout=0.0,
    )

    output = module(x)
    output.mean().backward()

    assert output.shape == (2, 7)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

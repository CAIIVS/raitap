"""Tests for method registry correctness"""

from __future__ import annotations

from raitap.transparency.methods import SHAP, Captum, ExplainerMethod


class TestMethodRegistry:
    """Test the method registry system"""

    def test_explainer_method_descriptor(self):
        """Test ExplainerMethod descriptor captures framework and algorithm"""
        method = Captum.IntegratedGradients
        assert isinstance(method, ExplainerMethod)
        assert method.framework == "captum"
        assert method.algorithm == "IntegratedGradients"

    def test_captum_methods_exist(self):
        """Test all documented Captum methods are registered"""
        assert hasattr(Captum, "IntegratedGradients")
        assert hasattr(Captum, "Saliency")
        assert hasattr(Captum, "LayerGradCam")  # Note: renamed from GradCAM
        assert hasattr(Captum, "DeepLift")
        assert hasattr(Captum, "GuidedBackprop")

    def test_shap_methods_exist(self):
        """Test all documented SHAP methods are registered"""
        assert hasattr(SHAP, "GradientExplainer")
        assert hasattr(SHAP, "DeepExplainer")
        assert hasattr(SHAP, "KernelExplainer")
        assert hasattr(SHAP, "TreeExplainer")

    def test_method_repr(self):
        """Test string representation"""
        method = Captum.Saliency
        assert "ExplainerMethod" in repr(method)
        assert "captum" in repr(method)
        assert "Saliency" in repr(method)

    def test_convenience_aliases(self):
        """Test convenience aliases work"""
        from raitap.transparency.methods import IntegratedGradients, KernelShap

        assert IntegratedGradients.framework == "captum"
        assert IntegratedGradients.algorithm == "IntegratedGradients"
        assert KernelShap.framework == "shap"
        assert KernelShap.algorithm == "KernelExplainer"

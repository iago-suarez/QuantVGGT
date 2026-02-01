# Copyright (c) Meta Platforms, Inc. and affiliates.
# Operator audit utilities for TFLite export compatibility.
#
# This module helps identify operations that may not be compatible
# with torch.export or TFLite conversion.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Set, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# Operations known to be problematic for TFLite export
PROBLEMATIC_OPS = {
    # Fused CUDA operations
    "scaled_dot_product_attention": "Use manual attention computation",
    "memory_efficient_attention": "Use manual attention computation", 
    "flash_attention": "Use manual attention computation",
    
    # Dynamic operations
    "torch.compile": "Remove torch.compile decorators",
    "torch.jit.script": "Use torch.jit.trace instead",
    
    # Unsupported quantization
    "fake_quant": "Use TFLite native quantization",
    "hadamard": "Custom operation not supported",
    
    # Control flow
    "torch.cond": "Use static control flow",
    "while_loop": "Unroll loops where possible",
}


class OperatorAuditor:
    """
    Audits a PyTorch model for TFLite export compatibility.
    
    Usage:
        auditor = OperatorAuditor()
        issues = auditor.audit(model, sample_input)
        auditor.print_report(issues)
    """
    
    def __init__(self):
        self.traced_ops: Set[str] = set()
        self.issues: List[Dict[str, Any]] = []
    
    def audit(
        self,
        model: nn.Module,
        sample_input: Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Audit model for export compatibility issues.
        
        Args:
            model: PyTorch model to audit
            sample_input: Sample input tensor
            
        Returns:
            List of issues found, each as a dict with 'op', 'location', 'suggestion'
        """
        self.issues = []
        
        # Check 1: Module-level issues
        self._audit_modules(model)
        
        # Check 2: Try torch.export and catch errors
        self._try_export(model, sample_input)
        
        # Check 3: Try JIT trace for additional info
        self._try_trace(model, sample_input)
        
        return self.issues
    
    def _audit_modules(self, model: nn.Module) -> None:
        """Check modules for known problematic patterns."""
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # Check for fused attention
            if hasattr(module, 'fused_attn') and module.fused_attn:
                self.issues.append({
                    'op': 'fused_attn',
                    'location': f"{name} ({module_type})",
                    'severity': 'ERROR',
                    'suggestion': 'Set fused_attn=False or use ExportableAttention',
                })
            
            # Check for dropout (should be disabled for export)
            if isinstance(module, nn.Dropout) and module.p > 0:
                self.issues.append({
                    'op': 'Dropout',
                    'location': f"{name}",
                    'severity': 'WARNING',
                    'suggestion': 'Set dropout to 0 or call model.eval()',
                })
            
            # Check for dynamic position caching
            if hasattr(module, 'position_cache') or hasattr(module, 'frequency_cache'):
                self.issues.append({
                    'op': 'dynamic_cache',
                    'location': f"{name} ({module_type})",
                    'severity': 'WARNING', 
                    'suggestion': 'Pre-compute positions for fixed input size',
                })
    
    def _try_export(self, model: nn.Module, sample_input: Tensor) -> None:
        """Try torch.export and capture any errors."""
        try:
            with torch.no_grad():
                _ = torch.export.export(model, (sample_input,))
            logger.info("torch.export succeeded")
        except Exception as e:
            error_msg = str(e)
            self.issues.append({
                'op': 'torch.export',
                'location': 'model',
                'severity': 'ERROR',
                'suggestion': f'Export failed: {error_msg[:200]}...',
            })
            
            # Parse error for specific ops
            for op, suggestion in PROBLEMATIC_OPS.items():
                if op.lower() in error_msg.lower():
                    self.issues.append({
                        'op': op,
                        'location': 'detected in error',
                        'severity': 'ERROR',
                        'suggestion': suggestion,
                    })
    
    def _try_trace(self, model: nn.Module, sample_input: Tensor) -> None:
        """Try torch.jit.trace for additional diagnostics."""
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, sample_input)
            
            # Analyze traced graph for ops
            graph_str = str(traced.graph)
            
            # Check for problematic patterns
            if 'scaled_dot_product_attention' in graph_str:
                self.issues.append({
                    'op': 'scaled_dot_product_attention',
                    'location': 'traced graph',
                    'severity': 'ERROR',
                    'suggestion': 'Replace with manual Q@K->softmax->@V',
                })
                
        except Exception as e:
            self.issues.append({
                'op': 'torch.jit.trace',
                'location': 'model',
                'severity': 'WARNING',
                'suggestion': f'Trace failed (may indicate dynamic ops): {str(e)[:100]}',
            })
    
    def print_report(self, issues: List[Dict[str, Any]] = None) -> None:
        """Print a formatted report of audit findings."""
        if issues is None:
            issues = self.issues
        
        if not issues:
            print("\n✓ No export compatibility issues found!")
            return
        
        print("\n" + "=" * 60)
        print("VGGT Export Compatibility Report")
        print("=" * 60)
        
        errors = [i for i in issues if i['severity'] == 'ERROR']
        warnings = [i for i in issues if i['severity'] == 'WARNING']
        
        if errors:
            print(f"\n🔴 ERRORS ({len(errors)}):")
            print("-" * 40)
            for issue in errors:
                print(f"  Op: {issue['op']}")
                print(f"  Location: {issue['location']}")
                print(f"  Fix: {issue['suggestion']}")
                print()
        
        if warnings:
            print(f"\n🟡 WARNINGS ({len(warnings)}):")
            print("-" * 40)
            for issue in warnings:
                print(f"  Op: {issue['op']}")
                print(f"  Location: {issue['location']}")
                print(f"  Fix: {issue['suggestion']}")
                print()
        
        print("=" * 60)
        print(f"Total: {len(errors)} errors, {len(warnings)} warnings")
        print("=" * 60)


def audit_model(
    model: nn.Module,
    sample_input: Tensor,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to audit a model for export compatibility.
    
    Args:
        model: Model to audit
        sample_input: Sample input tensor
        verbose: Whether to print report
        
    Returns:
        List of issues found
    """
    auditor = OperatorAuditor()
    issues = auditor.audit(model, sample_input)
    
    if verbose:
        auditor.print_report(issues)
    
    return issues


def check_tflite_op_compatibility(op_name: str) -> Tuple[bool, str]:
    """
    Check if a specific operation is TFLite compatible.
    
    Args:
        op_name: Name of the operation
        
    Returns:
        Tuple of (is_compatible, message)
    """
    # TFLite supported ops (subset)
    TFLITE_SUPPORTED = {
        'linear', 'matmul', 'conv2d', 'relu', 'gelu', 'softmax',
        'layer_norm', 'batch_norm', 'dropout', 'add', 'mul',
        'reshape', 'transpose', 'permute', 'view', 'cat', 'split',
        'mean', 'sum', 'max', 'min', 'sqrt', 'exp', 'log',
    }
    
    op_lower = op_name.lower().replace('_', '')
    
    for supported in TFLITE_SUPPORTED:
        if supported in op_lower:
            return True, f"'{op_name}' is generally supported"
    
    if op_name.lower() in PROBLEMATIC_OPS:
        return False, PROBLEMATIC_OPS[op_name.lower()]
    
    return False, f"'{op_name}' compatibility unknown - test with torch.export"

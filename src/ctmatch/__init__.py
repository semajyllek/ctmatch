from .config import PipeConfig

def __getattr__(name):
    if name == "CTMatch":
        from .matching.pipeline import CTMatch
        return CTMatch
    if name == "Evaluator":
        from .evaluation.evaluator import Evaluator
        return Evaluator
    if name == "EvaluatorConfig":
        from .evaluation.evaluator import EvaluatorConfig
        return EvaluatorConfig
    raise AttributeError(f"module 'ctmatch' has no attribute {name!r}")

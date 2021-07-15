"""Copyright (c) Facebook, Inc. and its affiliates."""
import abc
from typing import Any, Dict


class Exportable(abc.ABC):
    @abc.abstractmethod
    def export(self) -> Dict[str, Any]:
        pass

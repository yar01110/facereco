from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import  Iterator, Optional ,Union

    
@dataclass
class Data():


    personxyxy: np.ndarray
    personImg: np.ndarray
    trackId: np.ndarray
    
    state: Optional[str] = "faceNotApeared"
    

    facexyxy: Optional[np.ndarray] = None
    faceImg: Optional[np.ndarray] = None
 
    identification : Optional[Union[str, dict]] = None
    
    
    
    def __iter__(self) -> Iterator[Data]:
        """returns an iterator that yields the current instance of "Data" class"""
        return iter([self])
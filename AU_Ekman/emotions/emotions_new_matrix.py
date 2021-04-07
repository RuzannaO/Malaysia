from .. import utils
import cv2
import numpy as np
from ..au.parameters import *
from ..au.variables import *
import pandas as pd



def get_emotions(list_of_au_current_frame, combos_matrix):
	list_of_au_current_frame = pd.DataFrame(list_of_au_current_frame)
	combo_names = np.array(combos_matrix.columns.tolist())
	combos = combos_matrix - list_of_au_current_frame.values
	combos = combos.fillna(0)
	combos = np.abs(combos)
	combos = np.sum(combos.values, 0)
	combos_binary = np.where(combos == 0, True, False)
	combos = combo_names[combos_binary]

	return combos

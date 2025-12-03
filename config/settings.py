import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

price_col = 'clausura'

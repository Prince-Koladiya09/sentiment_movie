import os
import traceback

os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("--- Diagnostics Start ---")
try:
    import tensorflow as tf
    print("TensorFlow Version:", tf.__version__)
except Exception:
    print("TensorFlow import failed")
    traceback.print_exc()

try:
    import tf_keras
    print("tf-keras Version:", tf_keras.__version__)
except Exception:
    print("tf-keras import failed")
    traceback.print_exc()

try:
    import transformers
    print("Transformers Version:", transformers.__version__)
    from transformers.utils import is_tf_available
    print("is_tf_available():", is_tf_available())
except Exception:
    print("Transformers basic import failed")
    traceback.print_exc()

try:
    from transformers import TFDistilBertForSequenceClassification
    print("TFDistilBertForSequenceClassification import SUCCESS")
except ImportError as e:
    print("ImportError:", e)
    traceback.print_exc()
except Exception as e:
    print("Exception during import:", e)
    traceback.print_exc()
print("--- Diagnostics End ---")

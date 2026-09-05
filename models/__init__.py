# Model zoo. Each module defines a `Model` class with:
#   - Model.info: dict with 'name', 'file', 'ratio_support'
#   - Model.get_info() / get_name()
#   - Model.get_model(): inference model (finest pyramid level only)
#   - Model.get_training_model(): training model (returns all pyramid levels)
#
# Training scripts import these dynamically (find_and_import_model) so a new
# version is just a new file: flownetN_vNNN.py / warpnetN_vNNN.py.

class FeatureExtractor:
    def __init__(self, model, is_student=False):
        self.model = model
        self.is_student = is_student
        self.features = {}
        self.hooks = []
        if self.is_student:
            # CORRECTED MAPPING FOR YOUR 3-LAYER STUDENT
            self.target_layers = {
                'e_conv1': 'e_conv2',  # Student's 1st layer → Teacher's 2nd layer
                'e_conv2': 'e_conv4',  # Student's 2nd layer → Teacher's 4th layer
                'e_conv3': 'e_conv7'   # Student's 3rd layer → Teacher's final layer
            }
        else: # Teacher model
            self.target_layers = {
                'e_conv2': 'e_conv2_features',
                'e_conv4': 'e_conv4_features',
                'e_conv7': 'e_conv7_features',
            }

    def register_hooks(self):
        # Register hooks only for the layers specified in target_layers
        layer_names_to_hook = list(self.target_layers.keys())

        for name, module in self.model.named_modules():
             if name in layer_names_to_hook:
                 hook = module.register_forward_hook(self.save_features(name))
                 self.hooks.append(hook)

    def save_features(self, layer_name):
        def hook(module, input, output):
            if self.is_student:
                # Save student features with a key that indicates the corresponding teacher layer
                corresponding_teacher_layer = self.target_layers.get(layer_name, layer_name)
                feature_name = f"student_{layer_name}_matches_teacher_{corresponding_teacher_layer}"
            else:
                # Save teacher features with a descriptive name
                feature_name = self.target_layers[layer_name]
            self.features[feature_name] = output
        return hook

    def clear_features(self):
        self.features = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

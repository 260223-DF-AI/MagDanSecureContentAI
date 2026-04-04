
class ClassificationTraining():
    
    def __init__(self, id, confidence_score, pred_category, is_correct, image_key, epoch_key):
        self.id = int(id)
        self.confidence_score = float(confidence_score)
        self.pred_category = str(pred_category)
        self.is_correct = bool(is_correct)
        # FKs
        self.image_key = int(image_key)
        self.epoch_key = int(epoch_key)

    def _validate_types(self):
        # TODO: better validation for classes/cats; dim/images need this too
        if self.pred_category not in ['groups', 'men', 'women']:
            raise ValueError(f'Invalid category: {self.pred_category} | should be "groups", "men", or "women"')

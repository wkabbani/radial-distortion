from fastai.learner import load_learner

# this function should be in the same file as load_learner
def label_func(f): return 'undistorted' in str(f)

def get_label_text(lbl): return 'Undistorted' if lbl else 'Distorted'

def get_quality_value(learner, image_path):
    lbl, idx, props = learner.predict(image_path)
    quality_value = props[1].cpu().item()
    print(f'image: {image_path}, quality value: {quality_value}, label: {get_label_text(lbl)}')
    return quality_value


if __name__ == "__main__":

    image_path = 'path/to/image.jpg'
    ckp_path = 'path/to/model.pkl'

    learn = load_learner(ckp_path)
    get_quality_value(learn, image_path)

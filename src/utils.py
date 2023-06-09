import functools
import time
import numpy as np
import cv2
import config


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def prepare_face_bank(recognizer, tta=True):
    embeddings = []
    names = ['Unknown']
    for path in config.face_bank_path.iterdir():
        if path.is_dir():
            print(path)
            embedding = []
            for image_path in path.iterdir():
                try:
                    print(image_path)
                    image = cv2.imread(str(image_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    embs, _ = recognizer.get_emb(image_rgb)
                    embedding.append(embs[0])

                except Exception as e:
                    print(e)

            if len(embedding) > 0:
                embedding = np.array(embedding).mean(axis=0)
                embeddings.append(embedding)
                names.append(path.name)
    
    embeddings = np.array(embeddings)
    np.save('face_bank', {"embeddings": embeddings, "names": names})
    return embeddings, names


def load_face_bank():
    data = np.load('face_bank.npy', allow_pickle=True)
    embeddings = data.item().get('embeddings')
    names = data.item().get('names')
    return embeddings, names


def draw_box_name(image, bbox, name, kps=None):
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    image = cv2.putText(image, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    if kps is not None:
        for kp in kps:
            kp = kp.astype(int)
            cv2.circle(image, tuple(kp) , 1, (0,0,255) , 2)

    return image

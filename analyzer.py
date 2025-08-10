import os
import cv2
import torch
import numpy as np
from main import CNN
from glob import glob

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def draw(event, current_x, current_y, flags, params):
    global x, y, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x = current_x
        y = current_y
        
        drawing = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (current_x, current_y), (x, y), (255, 255, 255), thickness=1)
            x, y = current_x, current_y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def get_best_model():
    curr_dir = os.path.abspath(os.curdir)

    model_paths = glob(f'{curr_dir}/models/*')    

    best_path = ''
    best_avg = -1

    for path in model_paths:
        eval, train = float(path.split('-')[-2]), float(path.split('-')[-1][0:-3])

        if (eval + train) / 2 >= best_avg:
            best_path = path
            best_avg = (eval + train) / 2

    print(f'Best average: {best_avg} --- Best Model: {best_path}')

    return torch.load(best_path, weights_only=False)
        


if __name__ == '__main__':
    model = get_best_model()
    model.eval()
    model.to(device=DEVICE)

    canvas = np.ones((28, 28, 1), np.uint8)

    x = 0
    y = 0
    drawing = False

    cv2.imshow('Draw', canvas)

    cv2.setMouseCallback('Draw', draw)

    while True:
        cv2.imshow('Draw', canvas)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break

        elif cv2.waitKey(1) & 0xFF == ord('a'):
            print(f'analyzing....')
            image = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                image = image.to(DEVICE)
                outputs = model(image)
                predicted_class = outputs.argmax(dim=1).item()

            print(f"Predicted digit: {predicted_class}")
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            canvas = np.ones((28, 28, 1), np.uint8)
        

    cv2.destroyAllWindows()
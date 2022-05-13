import argparse

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='IO/input/test.jpg', help='Image file')
    parser.add_argument('--config', type=str, default='configs/scrfd/scrfd_500m_bnkps.py', help='Config file')
    parser.add_argument('--checkpoint', type=str, default='weights/SCRFD_500M_KPS.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    print(result)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()

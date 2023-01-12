import edgeiq
import time


def main():
    instance_segmentation = edgeiq.InstanceSegmentation("alwaysai/mask_rcnn")
    if edgeiq.is_opencv_cuda_available():
        engine = edgeiq.Engine.DNN_CUDA
    else:
        engine = edgeiq.Engine.DNN
    instance_segmentation.load(engine)

    print("Engine: {}".format(instance_segmentation.engine))
    print("Accelerator: {}\n".format(instance_segmentation.accelerator))
    print("Model:\n{}\n".format(instance_segmentation.model_id))
    print("Labels:\n{}\n".format(instance_segmentation.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.FileVideoStream('videos/sample.mkv') as video_stream, \
                edgeiq.Streamer() as streamer:
            time.sleep(2.0)
            fps.start()

            while True:
                try:
                    frame = video_stream.read()
                except edgeiq.NoMoreFrames:
                    video_stream.start()
                    frame = video_stream.read()

                results = instance_segmentation.segment_image(frame)

                # Generate text to display on streamer
                text = ["Model: {}".format(instance_segmentation.model_id)]

                text.append("Inference time: {:1.3f} s".
                            format(results.duration))

                frame = instance_segmentation.markup_image(frame,
                                                           results.predictions)

                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))
        print("Program Ending")


if __name__ == "__main__":
    main()

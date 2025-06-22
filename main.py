from segment_teeth import YoloToothSegmenter
from plotting_exmaples import plot_whole_image_with_mask, plot_one_tooth_test



def run_segmentation():
    segmenter = YoloToothSegmenter("default.yaml")
    segmenter.run()


def main():
   #run_segmentation()
   #plot_one_tooth_test()
   #plot_unet_predictions()
   plot_whole_image_with_mask()

if __name__ == "__main__":
    main()

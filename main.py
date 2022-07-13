import tensor_check
# Import script of tensor part 00
from tensor_00_fundamental import tensor_00_0_creating, tensor_00_1_get_info
from tensor_00_fundamental import tensor_00_2_manipulating, tensor_00_3_function
from tensor_00_fundamental import tensor_00_4_exercise
# Import script of tensor part 01
from tensor_01_regression import tensor_01_0_beginning, tensor_01_1_modeling
from tensor_01_regression import tensor_01_2_visualizing, tensor_01_3_improving_model
from tensor_01_regression import tensor_01_4_process_model, tensor_01_5_larger_example
from tensor_01_regression import tensor_01_6_exercises
# Import script of tensor part 02
from tensor_02_classification import tensor_02_0_beginning, tensor_02_1_modeling
from tensor_02_classification import tensor_02_2_non_linearity, tensor_02_3_eval_and_improve
from tensor_02_classification import tensor_02_4_larger_example, tensor_02_5_exercises
# Import script of tensor part 03
from tensor_03_computer_vision import tensor_03_0_beginning, tensor_03_1_binary_classification
from tensor_03_computer_vision import tensor_03_2_multi_class_classification, tensor_03_3_exercises
# Import script of tensor part 04
from tensor_04_transfer_learning.tensor_04_part_1_feature_extraction import tensor_04_p1_0_beginning


def main():
    """Main Script for running all other parts of my curriculum"""

    # script for checking the TensorFlow version & GPU availability
    # tensor_check.run()

    # script for tensor part 00 of fundamental
    # tensor_00_0_creating.run()
    # tensor_00_1_get_info.run()
    # tensor_00_2_manipulating.run()
    # tensor_00_3_function.run()
    # tensor_00_4_exercise.run()

    # script for tensor part 01 of regression
    # tensor_01_0_beginning.run()
    # tensor_01_1_modeling.run()
    # tensor_01_2_visualizing.run()
    # tensor_01_3_improving_model.run()
    # tensor_01_4_process_model.run()
    # tensor_01_5_larger_example.run()
    # tensor_01_6_exercise.run()

    # script for tensor part 02 of classification
    # tensor_02_0_beginning.run()
    # tensor_02_1_modeling.run()
    # tensor_02_2_non_linearity.run()
    # tensor_02_3_eval_and_improve.run()
    # tensor_02_4_larger_example.run()
    # tensor_02_5_exercises.run()

    # script for tensor part 03 of computer vision classification
    # tensor_03_0_beginning.run()
    # tensor_03_1_binary_classification.run()
    # tensor_03_2_multi_class_classification.run()
    # tensor_03_3_exercises.run()

    # script for tensor part 04 of transfer learning part 1: Feature Extraction
    tensor_04_p1_0_beginning.run()


if __name__ == "__main__":
    main()

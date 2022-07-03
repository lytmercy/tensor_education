import tensor_check
# Import script of tensor part 00
from tensor_00_fundamental import tensor_00_0_creating, tensor_00_1_get_info
from tensor_00_fundamental import tensor_00_2_manipulating, tensor_00_3_function
from tensor_00_fundamental import tensor_00_4_exercise
# Import script of tensor part 01
from tensor_01_regression import tensor_01_0_beginning, tensor_01_1_modeling
from tensor_01_regression import tensor_01_2_visualizing, tensor_01_3_improving_model
from tensor_01_regression import tensor_01_4_process_model, tensor_01_5_larger_example
from tensor_01_regression import tensor_01_6_exercise
# Import script of tensor part 02
from tensor_02_classification import tensor_02_0_beginning


def main():
    """Main Script for running all other parts of my curriculum"""

    # script for checking the TensorFlow version & GPU availability
    # tensor_check.run()

    # script for tensor part 00 of curriculum
    # tensor_00_0_creating.run()
    # tensor_00_1_get_info.run()
    # tensor_00_2_manipulating.run()
    # tensor_00_3_function.run()
    # tensor_00_4_exercise.run()

    # script for tensor part 01 of curriculum
    # tensor_01_0_beginning.run()
    # tensor_01_1_modeling.run()
    # tensor_01_2_visualizing.run()
    # tensor_01_3_improving_model.run()
    # tensor_01_4_process_model.run()
    # tensor_01_5_larger_example.run()
    # tensor_01_6_exercise.run()

    # script for tensor part 02 of curriculum
    tensor_02_0_beginning.run()


if __name__ == "__main__":
    main()

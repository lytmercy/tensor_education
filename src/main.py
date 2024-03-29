# Import script for check TensorFlow version and GPU availability
import tensor_check
# Import scripts of tensor 00
from src.tensor_00_fundamental import tensor_00_0_creating, tensor_00_1_get_info, tensor_00_2_manipulating, \
                                      tensor_00_3_function, tensor_00_4_exercise
# Import scripts of tensor 01
from src.tensor_01_regression import tensor_01_0_beginning, tensor_01_1_modeling, tensor_01_2_visualizing, \
                                     tensor_01_3_improving_model, tensor_01_4_process_model,\
                                     tensor_01_5_larger_example, tensor_01_6_exercises
# Import scripts of tensor 02
from src.tensor_02_classification import tensor_02_0_beginning, tensor_02_1_modeling, tensor_02_2_non_linearity,\
                                         tensor_02_3_eval_and_improve, tensor_02_4_larger_example, tensor_02_5_exercises
# Import scripts of tensor 03
from src.tensor_03_computer_vision import tensor_03_0_beginning, tensor_03_1_binary_classification, \
                                          tensor_03_2_multi_class_classification, tensor_03_3_exercises
# Import scripts of tensor 04 part 1
from src.tensor_04_transfer_learning.tensor_04_part_1_feature_extraction import tensor_04_p1_0_beginning, \
                                                                                tensor_04_p1_1_tensorhub,\
                                                                                tensor_04_p1_2_exercises
# Import scripts of tensor 04 part 2
from src.tensor_04_transfer_learning.tensor_04_part_2_fine_tuning import tensor_04_p2_0_beginning, \
                                                                         tensor_04_p2_1_series_transfer_learning,\
                                                                         tensor_04_p2_2_exercises
# Import scripts of tensor 04 part 3
from src.tensor_04_transfer_learning.tensor_04_part_3_scaling_up import tensor_04_p3_0_beginning, \
                                                                        tensor_04_p3_1_big_dog_model,\
                                                                        tensor_04_p3_2_exercises
# Import scripts of tensor 07
from src.tensor_07_milestone_project_1 import tensor_07_0_preprocess_data, tensor_07_1_modeling,\
                                              tensor_07_2_todo_tasks, tensor_07_3_exercises
# Import scripts of tensor 08
from src.tensor_08_nlp_with_tf import tensor_08_0_preprocess_data, tensor_08_1_modeling, tensor_08_2_exercises


def main():
    """Main Script for running all other parts of my curriculum"""

    # script for checking the TensorFlow version & GPU availability
    # tensor_check.run()

    # scripts of tensor 00 of TensorFlow fundamentals
    # tensor_00_0_creating.run()
    # tensor_00_1_get_info.run()
    # tensor_00_2_manipulating.run()
    # tensor_00_3_function.run()
    # tensor_00_4_exercise.run()

    # scripts of tensor 01 of regression with TensorFlow
    # tensor_01_0_beginning.run()
    # tensor_01_1_modeling.run()
    # tensor_01_2_visualizing.run()
    # tensor_01_3_improving_model.run()
    # tensor_01_4_process_model.run()
    # tensor_01_5_larger_example.run()
    # tensor_01_6_exercise.run()

    # scripts of tensor 02 of classification with TensorFlow
    # tensor_02_0_beginning.run()
    # tensor_02_1_modeling.run()
    # tensor_02_2_non_linearity.run()
    # tensor_02_3_eval_and_improve.run()
    # tensor_02_4_larger_example.run()
    # tensor_02_5_exercises.run()

    # script of tensor 03 of computer vision classification with TensorFlow
    # tensor_03_0_beginning.run()
    # tensor_03_1_binary_classification.run()
    # tensor_03_2_multi_class_classification.run()
    # tensor_03_3_exercises.run()

    # script of tensor 04 of transfer learning part 1: Feature Extraction
    # tensor_04_p1_0_beginning.run()
    # tensor_04_p1_1_tensorhub.run()
    # tensor_04_p1_2_exercise.run()

    # script of tensor 04 of transfer learning part 2: Fine-tuning
    # tensor_04_p2_0_beginning.run()
    # tensor_04_p2_1_series_transfer_learning.run()
    # tensor_04_p2_2_exercises.run()

    # scripts of tensor 04 of transfer learning part 3: Scaling up
    # tensor_04_p3_0_beginning.run()
    # tensor_04_p3_1_big_dog_model.run()
    # tensor_04_p3_2_exercises.run()

    # scripts of tensor 07 of milestone project_1: Food Vision
    # tensor_07_0_preprocess_data.run()
    # tensor_07_1_modeling.run()
    # tensor_07_2_todo_tasks.run()
    # tensor_07_2_exercises.run()

    # scripts of tensor 08 of Natural Language Processing with TensorFlow
    # tensor_08_0_preprocess_data.run()
    # tensor_08_1_modeling.run()
    tensor_08_2_exercises.run()


if __name__ == "__main__":
    main()


from models.model_5.predict_model import predict_model as predict_model_5
from models.model_cv.predict_model import predict_model as predict_model_cv
from models.model_10.predict_model import predict_model as predict_model_10
from models.model_10.train_model import train as train_model_10
from models.model_5.train_model import train as train_model_5
from models.model_cv.train_model import train as train_model_CV


def main():

    cli_init()


def cli_init():
    """ Command line interface initialization.
    """
    print("Serotonin Level Detection")
    print("Select function (e.g. Train): ")
    print(" - Train")
    print(" - Predict")

    function_input = input("Enter function: ")
    if function_input.lower() == "train":
        print("Selected function: Train")
        print("Enter data type (select number): ")
        print("1. 10 second colour plots")
        print("2. 5 second colour plots (Disabled)")
        print("3. CVs")
        training_type = input("")
        if training_type == "1":
            print("Selected data type: 10 second colour plots")
            cross_val = select_cross_val()
            train_model_10(select_version(), cross_val)
        elif training_type == "2":
            print("Selected data type: 5 second colour plots")
            cross_val = select_cross_val()
            train_model_5(select_version(), cross_val)
        elif training_type == "3":
            print("Selected data type: CVs")
            cross_val = select_cross_val()
            train_model_CV(select_version(), cross_val)
        else:
            print("Invalid input")
    elif function_input.lower() == "predict":
        print("Selected function: Predict")
        print("Enter data type (select number): ")
        print("1. 10 second colour plots")
        print("2. 5 second colour plots")
        print("3. CVs")
        training_type = input("")
        if training_type == "1":
            print("Selected data type: 10 second colour plots")
            predict_model_10("../models/model_10/v5.0/version_5.0_10sec_best_model.h5",
                             "../data/external/comparison_dataset/10s")
        elif training_type == "2":
            print("In development...")
            print("Selected data type: 5 second colour plots")
            predict_model_5("../models/model_5/vfinal/version_final_5sec_best_model.h5",
                            "../data/external/comparison_dataset/5s")
        elif training_type == "3":
            print("Selected data type: CVs")
            predict_model_cv("../models/model_cv/vfinal/version_final_consecutive_best_model.h5",
                             "../data/external/comparison_dataset/cvs")
        else:
            print("Invalid input")
    else:
        print("Invalid input")


def select_version():
    version = input("Enter model version: (e.g 1.1.2)")

    return version


def select_cross_val():
    print("Is 5-fold cross validation required?")
    print("1. Yes")
    print("2. No")
    cross_val = input("")
    if cross_val == "1":
        return True

    return False


if __name__ == "__main__":
    main()

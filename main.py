from naive_bayes import NaiveBayes

if __name__ == "__main__":
    file_name = "cardaten/car.data"
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    target = 'acceptability'



    # run the classifier
    sum_of_errors = 0
    lowest_error = 0
    highest_error = 0

    for i in range(0, 100):
        print("Iteration ", i+1)
        # initialize a NaiveBayes object with the file, attributes, and target attribute
        naive_bayes = NaiveBayes(file_name, attributes, target)

        flag = False
        if i == 99:
            flag = True
        sum_of_errors += naive_bayes.execute(flag)

    mean_error = sum_of_errors/100

    error_rate = (mean_error/len(naive_bayes.test_data)) * 100

    print("Mean Error Rate: ", error_rate)

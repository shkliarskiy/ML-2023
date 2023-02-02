def check_result(grader_obj):
    def wrapper():
        if len((grader_obj.check('q1').failed_tests)) == 0:
            print('Завдання виконано успішно!')
        else:
            print('В завданні присутні помилки',
            grader_obj.check('q1').failed_tests)
    return wrapper

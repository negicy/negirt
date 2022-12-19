

def test_DI_make_candidate(q_data, qualify_task):
    if len(q_data) == len(qualify_task):
        message = 'test passed!'
    else:
        message = "test failed !"
    return message

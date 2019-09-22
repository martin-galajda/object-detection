def assert_true(cond_result: bool, err_msg: str):
    if not cond_result:
        raise RuntimeError(f'Expected to be true: {err_msg}')

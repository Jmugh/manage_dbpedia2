'''
返回当前时间  年_月_日_时_分_秒
'''
def get_present_time()->str:
    import time
    return time.strftime("%Y%m%d  %H_%M")
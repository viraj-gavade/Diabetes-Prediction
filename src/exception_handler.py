import sys
from src.logger import logging


def error_message_details(error,error_details:sys):
    _,_,error_tab = error_details.exc_info()
    error_filename = error_tab.tb_frame.f_code.co_filename
    error_line_no = error_tab.tb_lineno
    error_message = f'Error Occured in File : {error_filename} \n Error Line : {error_line_no}'
    return error_message


class CustomException(Exception):
    def __init__(self, error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_details)


        def __str__(self):
            return self.error_message
    

if __name__ == "__main__":
    try:
        a=1/0
    except Exception as e:
        print(e)
        logging.info('Divide by zero error')
        raise CustomException(e,sys)
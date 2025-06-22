import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):   # Function to extract detailed error message
    _,_,exc_tb=error_detail.exc_info()   # Get the traceback details
    file_name=exc_tb.tb_frame.f_code.co_filename  # Get the name of the file where the error occurred
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))    # Format the error message with file name, line number, and error message 

    return error_message    

    

class CustomException(Exception):  # Custom exception class to handle exceptions with detailed error messages
    def __init__(self,error_message,error_detail:sys):  # Initialize the exception with an error message and error details
        super().__init__(error_message)  # Call the parent class constructor
        self.error_message=error_message_detail(error_message,error_detail=error_detail)  # Extract detailed error message using the helper function
    
    def __str__(self):  # String representation of the exception
        return self.error_message
    


        
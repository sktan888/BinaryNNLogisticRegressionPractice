import logging


def log(msg):
    """
    This function records log messages into a file 'mylog.log'

    Argument:


    Returns:

    """

    logging.basicConfig(
        filename="mylog.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.debug(msg)

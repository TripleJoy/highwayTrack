import logging


def get_logger(logger_name):
    logging.basicConfig(
        format='%(asctime)s [%(name)s]-[%(levelname)s]: %(message)s',
        datefmt='%H:%M:%S',level=logging.INFO)
    logger = logging.getLogger(logger_name)

    return logger


if __name__ == '__main__':
    lg = get_logger('1')
    lg.info('???')


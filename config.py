import configparser

_config = configparser.ConfigParser()
_config.read("config.conf")

database = {
    "confield": "database",
    "host": _config.get("database", "host"),
    "port": _config.get("database", "port"),
    "username": _config.get("database", "username"),
    "password": _config.get("database", "password"),
    "dbname": _config.get("database", "dbname"),
}

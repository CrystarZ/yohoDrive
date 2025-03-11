import configparser


class decoder:
    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.parser = configparser.ConfigParser()
        self.parser.read(path, encoding)
        pass

    @property
    def dict(self):
        d = {}
        for section in self.sections:
            d[section] = dict(self.parser.items(section))
        return d

    @property
    def sections(self) -> list[str]:
        return self.parser.sections()

    def Section(self, section: str):
        return Section(self, section)

    def get(self, section: str, option: str):
        return self.parser.get(section, option)


class Section:
    def __init__(self, decoder: decoder, section: str) -> None:
        self.decoder = decoder
        self.section = section
        pass

    @property
    def dict(self) -> dict:
        d = dict(self.decoder.parser.items(self.section))
        # d["_section"] = self.section
        return d

    def get(self, option: str):
        return self.decoder.get(self.section, option)

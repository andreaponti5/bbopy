from typing import List, Optional, Union

from bbopy.utils.verbose import Table


class Display:

    def __init__(
            self,
            title: str = "Experiment",
            columns: Optional[List[str]] = None,
            widths: Optional[List[int]] = None,
    ) -> None:
        if columns is None:
            columns = ["#", "N. Eval.", "Best Seen", "Time"]
        if widths is None:
            widths = [8, 10, 15, 15, 8]
        self.table = Table(columns, widths, title=title)

    def update(self, row: List[Union[int, float, str]]) -> None:
        self.table.append(row)

    def println(self, index: Optional[int] = None):
        if index is None:
            index = -1
        print(self.table[index])

    def print_header(self):
        print(self.table["title"])
        print(self.table["header"])

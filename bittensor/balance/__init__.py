class Balance:
    unit = "Tao"
    rao : int
    tao: float

    def __init__(self, balance):
        self.planck_tao = balance
        self.tao = self.planck_tao / pow(10, 9)

    def __str__(self):
        return "{unit:s} {balance:.9f}".format(unit=self.unit, balance=self.tao)

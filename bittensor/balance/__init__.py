class Balance:
    unit = "TAO"
    planck_tao : int
    tao: float

    def __init__(self, balance):
        self.planck_tao = balance
        self.tao = self.planck_tao / pow(10, 15)

    def __str__(self):
        return "{unit:s} {balance:.15f}".format(unit=self.unit, balance=self.tao)

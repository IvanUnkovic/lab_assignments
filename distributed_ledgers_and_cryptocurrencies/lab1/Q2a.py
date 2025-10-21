from sys import exit
from bitcoin.core.script import OP_DUP, OP_ADD, OP_EQUALVERIFY, OP_SUB, OP_EQUAL, OP_SWAP

from lib.utils import *
from lib.config import (my_private_key, my_public_key, my_address,
                    faucet_address, network_type)
from Q1 import send_from_P2PKH_transaction


######################################################################
# TODO: Implementirajte `scriptPubKey` za zadatak 2
Q2a_txout_scriptPubKey = [
    OP_ADD,
    0xe44,
    OP_EQUALVERIFY,
    OP_SUB,
    0x362,
    OP_EQUAL
]
######################################################################

if __name__ == '__main__':
    ######################################################################
    # Postavite parametre transakcije
    # TODO: amount_to_send = {cjelokupni iznos BCY-a u UTXO-u kojeg saljemo} - {fee}
    amount_to_send = 0.000165 - 0.00001
    # TODO: Identifikator transakcije
    txid_to_spend = (
        'de89ce2d58f50c449e16f494cdc3106dec0e01f84343c58f434814239cc795cc')
    # TODO: indeks UTXO-a unutar transakcije na koju se referiramo
    # (indeksi pocinju od nula)
    utxo_index = 1
    ######################################################################

    response = send_from_P2PKH_transaction(
        amount_to_send, txid_to_spend, utxo_index,
        Q2a_txout_scriptPubKey, my_private_key, network_type)
    print(response.status_code, response.reason)
    print(response.text)

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./BoxOracle.sol";

contract Betting {

    struct Player {
        uint8 id;
        string name;
        uint totalBetAmount;
        uint currCoef; 
    }
    struct Bet {
        address bettor;
        uint amount;
        uint player_id;
        uint betCoef;
    }

    address private betMaker;
    BoxOracle public oracle;
    uint public minBetAmount;
    uint public maxBetAmount;
    uint public totalBetAmount;
    uint public thresholdAmount;

    Bet[] private bets;
    Player public player_1;
    Player public player_2;

    bool private suspended = false;
    mapping (address => uint) public balances;
    
    constructor(
        address _betMaker,
        string memory _player_1,
        string memory _player_2,
        uint _minBetAmount,
        uint _maxBetAmount,
        uint _thresholdAmount,
        BoxOracle _oracle
    ) {
        betMaker = (_betMaker == address(0) ? msg.sender : _betMaker);
        player_1 = Player(1, _player_1, 0, 200);
        player_2 = Player(2, _player_2, 0, 200);
        minBetAmount = _minBetAmount;
        maxBetAmount = _maxBetAmount;
        thresholdAmount = _thresholdAmount;
        oracle = _oracle;

        totalBetAmount = 0;
    }

    receive() external payable {}

    fallback() external payable {}
    
    function makeBet(uint8 _playerId) public payable {
        //TODO Your code here
        require(!suspended); //meč ne smije biti suspendiran
        require(oracle.getWinner() == 0); //provjerava da nije zavrsila tekma
        require(msg.sender != betMaker);
        require(msg.value >= minBetAmount);
        require(msg.value <= maxBetAmount);
        require(_playerId == player_1.id || _playerId == player_2.id); //smiješ se kladiti na samo dva id-a moguća

        Player storage player = _playerId == player_1.id ? player_1 : player_2; //stvaramo playera
        player.totalBetAmount += msg.value; //ukupni betovi za tog playera povecani za bet
        bets.push(Bet(msg.sender, msg.value, _playerId, player.currCoef)); 
        totalBetAmount += msg.value; 

        if (totalBetAmount > thresholdAmount){
            if (player_1.totalBetAmount == totalBetAmount || player_2.totalBetAmount == totalBetAmount){
                suspended = true;
            } else {
                //solidity ne podrzava floating point operacije, moramo mnoziti sa 100
                player_1.currCoef = (player_1.totalBetAmount + player_2.totalBetAmount) * 100 / player_1.totalBetAmount;
                player_2.currCoef = (player_1.totalBetAmount + player_2.totalBetAmount) * 100 / player_2.totalBetAmount;
            }
        }
    }

    function claimSuspendedBets() public {
        //TODO Your code here
        require(suspended);

        uint moneyToReturn = 0;
        for (uint i=0; i<bets.length;){
            if(bets[i].bettor == msg.sender){
                moneyToReturn += bets[i].amount;
                bets[i] = bets[bets.length - 1];
                bets.pop();
            } else {
                i++;
            }

        }

        require(moneyToReturn > 0);
        payable(msg.sender).transfer(moneyToReturn);
    }
    
    function claimWinningBets() public {
        //TODO Your code here
        require(!suspended);
        require(oracle.getWinner() != 0);

        uint idWinner = oracle.getWinner();
        uint winnings = 0;

        for (uint i=0; i<bets.length;){
            if(bets[i].bettor == msg.sender && bets[i].player_id == idWinner){
                winnings += (bets[i].amount * bets[i].betCoef)/100;
                bets[i] = bets[bets.length - 1];
                bets.pop();
            } else {
                i++;
            }
        }

        payable(msg.sender).transfer(winnings);

    }

    function claimLosingBets() public {
        // TODO Your code here
        require(msg.sender == betMaker);
        require(oracle.getWinner() !=0);

        uint idWinner = oracle.getWinner();
        bool stillToClaim = false;
        for (uint i=0; i<bets.length; i++){
            if (bets[i].player_id == idWinner){
                stillToClaim = true;
            }
        }

        require(!stillToClaim);  //betmaker ne mozete uzeti tude gubitke sve dok svi pobjednici nisu isplatili novac

        uint totalLost = address(this).balance;
        payable(betMaker).transfer(totalLost);
        selfdestruct(payable(betMaker)); //uništavanje pametnog ugovora nakon isplate
    }
}
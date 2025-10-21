// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Auction.sol";

contract EnglishAuction is Auction {

    uint internal highestBid;
    uint internal initialPrice;
    uint internal biddingPeriod;
    uint internal lastBidTimestamp;
    uint internal minimumPriceIncrement;

    address internal highestBidder;

    constructor(
        address _sellerAddress,
        address _judgeAddress,
        Timer _timer,
        uint _initialPrice,
        uint _biddingPeriod,
        uint _minimumPriceIncrement
    ) Auction(_sellerAddress, _judgeAddress, _timer) {
        initialPrice = _initialPrice;
        biddingPeriod = _biddingPeriod;
        minimumPriceIncrement = _minimumPriceIncrement;

        // Start the auction at contract creation.
        lastBidTimestamp = time();
    }
    
    function bid() public payable {
        // TODO Your code here
        require(outcome == Outcome.NOT_FINISHED);
        require(time() < (lastBidTimestamp + biddingPeriod)); //izmedu dva bida ne smije proci vise od bidding period

        uint minBid = highestBid == 0 ? initialPrice : (highestBid + minimumPriceIncrement);
        require(msg.value >= minBid);

        if (highestBid > 0) {
            payable(highestBidder).transfer(highestBid);
        }

        highestBid = msg.value;
        highestBidder = msg.sender;

        lastBidTimestamp = time();
    }

    function getHighestBidder() override public view returns (address) {
        // TODO Your code here
        if (time() < (lastBidTimestamp + biddingPeriod)) {
            return address(0); //ako još postoji mogućnost bidanja, onda vraca 0
        }
        //može se dogodit da nitko nije dao bid
        if (highestBid >= initialPrice){
            return highestBidder;
        
        } else {
            return address(0);
        }
    }

    function enableRefunds() public {
        // TODO Your code here
        require(outcome == Outcome.NOT_FINISHED);
        require(time() > (lastBidTimestamp + biddingPeriod));

        finishAuction(Outcome.NOT_SUCCESSFUL, address(0));
    }

}
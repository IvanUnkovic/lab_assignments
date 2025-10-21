// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Auction.sol";

contract DutchAuction is Auction {

    uint public initialPrice;
    uint public biddingPeriod;
    uint public priceDecrement;

    uint internal auctionEnd;
    uint internal auctionStart;

    /// Creates the DutchAuction contract.
    ///
    /// @param _sellerAddress Address of the seller.
    /// @param _judgeAddress Address of the judge.
    /// @param _timer Timer reference
    /// @param _initialPrice Start price of dutch auction.
    /// @param _biddingPeriod Number of time units this auction lasts.
    /// @param _priceDecrement Rate at which price is lowered for each time unit
    ///                        following linear decay rule.
    constructor(
        address _sellerAddress,
        address _judgeAddress,
        Timer _timer,
        uint _initialPrice,
        uint _biddingPeriod,
        uint _priceDecrement
    )  Auction(_sellerAddress, _judgeAddress, _timer) {
        initialPrice = _initialPrice;
        biddingPeriod = _biddingPeriod;
        priceDecrement = _priceDecrement;
        auctionStart = time();
        // Here we take light assumption that time is monotone
        auctionEnd = auctionStart + _biddingPeriod;
    }

    /// In Dutch auction, winner is the first pearson who bids with
    /// bid that is higher than the current prices.
    /// This method should be only called while the auction is active.
    function bid() public payable {
        // TODO Your code here
        require(outcome == Outcome.NOT_FINISHED);
        require(time() < auctionEnd);

        uint timeFromAuctionStart = time() - auctionStart;
        uint priceReduction = priceDecrement * timeFromAuctionStart;

        uint newPrice = initialPrice - priceReduction;

        require(newPrice > 0);
        require(msg.value >= newPrice);

        uint moneyToReturn = msg.value - newPrice;

        finishAuction(Outcome.SUCCESSFUL, msg.sender);

        if (moneyToReturn > 0) {
            payable(msg.sender).transfer(moneyToReturn);
        }
    }

    function enableRefunds() public {
        // TODO Your code here
        require(outcome == Outcome.NOT_FINISHED);
        require(time() >= auctionEnd);

        finishAuction(Outcome.NOT_SUCCESSFUL, address(0));
    }
}
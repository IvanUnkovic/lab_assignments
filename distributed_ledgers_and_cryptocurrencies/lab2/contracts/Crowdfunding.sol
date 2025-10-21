// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Timer.sol";

/// This contract represents most simple crowdfunding campaign.
/// This contract does not protects investors from not receiving goods
/// they were promised from crowdfunding owner. This kind of contract
/// might be suitable for campaigns that does not promise anything to the
/// investors except that they will start working on some project.
/// (e.g. almost all blockchain spinoffs.)
contract Crowdfunding {

    address private owner;

    Timer private timer;

    uint256 public goal;

    uint256 public endTimestamp;

    mapping (address => uint256) public investments;

    constructor(
        address _owner,
        Timer _timer,
        uint256 _goal,
        uint256 _endTimestamp
    ) {
        owner = (_owner == address(0) ? msg.sender : _owner);
        timer = _timer; // Not checking if this is correctly injected.
        goal = _goal;
        endTimestamp = _endTimestamp;
    }

    function invest() public payable {
        // TODO Your code here
        require(timer.getTime() < endTimestamp);
        require(msg.value > 0);
        investments[msg.sender] += msg.value;
    }

    function claimFunds() public {
        // TODO Your code here
        require(msg.sender == owner);
        require(timer.getTime() >= endTimestamp);
        require(address(this).balance >= goal);
        payable(owner).transfer(address(this).balance);
    }

    function refund() public {
        // TODO Your code here
        require(timer.getTime() >= endTimestamp);
        require(address(this).balance < goal);

        uint256 investmentsFromSender = investments[msg.sender];
        require(investmentsFromSender > 0);

        investments[msg.sender] = 0;
        payable(msg.sender).transfer(investmentsFromSender);        
    }
    
}
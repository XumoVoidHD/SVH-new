from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import TickerId
import threading
import time
import pandas as pd
from ibapi.order import Order
import logging

# Disable IBKR API verbose logging
logging.getLogger('ibapi').setLevel(logging.CRITICAL)


class IBBroker(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.prices = {}
        self.historical_data = {}
        self.order_status = {}
        self.lock = threading.Lock()
        self.connected = False
        self.next_valid_id = None
        # Set minimum API version to support fractional shares (version 163+)
        self.minVersion = 163

    def connect_to_ibkr(self, host="127.0.0.1", port=7497, client_id=1):
        """Connect to IBKR"""
        self.connect(host, port, client_id)
        
        # Start API thread
        api_thread = threading.Thread(target=self.run, daemon=True)
        api_thread.start()
        time.sleep(1)
        self.connected = True
        print("Connected to IBKR")

    def tickPrice(self, reqId: TickerId, tickType, price: float, attrib):
        if price > 0:
            with self.lock:
                self.prices[reqId] = price

    def historicalData(self, reqId, bar):
        with self.lock:
            if reqId not in self.historical_data:
                self.historical_data[reqId] = []
            self.historical_data[reqId].append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })

    def historicalDataEnd(self, reqId, start, end):
        print(f"ReqId {reqId}: Historical data download complete.")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Order status callback"""
        with self.lock:
            if orderId not in self.order_status:
                self.order_status[orderId] = {}
            self.order_status[orderId].update({
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avgFillPrice': avgFillPrice,
                'lastFillPrice': lastFillPrice
            })

    def openOrder(self, orderId, contract, order, orderState):
        """Open order callback"""
        with self.lock:
            if orderId not in self.order_status:
                self.order_status[orderId] = {}
            self.order_status[orderId].update({
                'contract': contract,
                'order': order,
                'orderState': orderState
            })

    def nextValidId(self, orderId):
        """Next valid order ID callback from IBKR"""
        with self.lock:
            self.next_valid_id = orderId
            print(f"Next valid order ID from IBKR: {orderId}")

    def error(self, reqId, errorCode, errorString):
        """Error callback - filter out common non-critical messages"""
        # Filter out common connection status messages that are not actual errors
        ignore_codes = {
            2104,  # Market data farm connection is OK
            2106,  # HMDS data farm connection is OK  
            2158,  # Sec-def data farm connection is OK
            2103,  # Market data farm connection is broken (temporary)
            2176,  # Fractional share warning (we handle this)
            2105,  # HMDS data farm connection is broken (temporary)
            2107,  # Market data farm connection is OK (alternative)
            2159,  # Sec-def data farm connection is OK (alternative)
        }
        
        if errorCode in ignore_codes:
            # These are just status messages, not real errors
            return
        
        # Only print actual errors
        print(f"IBKR Error {errorCode}: {errorString}")

    def stock_contract(self, symbol: str):
        """Create stock contract"""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def tickPrice(self, reqId, tickType, price, attrib):
        with self.lock:
            if tickType == 4:  # Last traded price
                self.prices[reqId] = price


    def get_current_price(self, symbol, reqId):
        """Get current price for symbol (last traded price)"""
        contract = self.stock_contract(symbol)
        self.reqMktData(reqId, contract, "", False, False, [])

        for _ in range(20):  # wait up to ~10s
            time.sleep(0.5)
            with self.lock:
                price_data = self.prices.get(reqId)
                if price_data is not None:  # float = last price
                    self.cancelMktData(reqId)
                    return price_data

        self.cancelMktData(reqId)
        return None



    def get_historical_data(self, symbol, reqId, duration="3 D", bar_size="3 mins"):
        """Get historical data for symbol"""
        contract = self.stock_contract(symbol)
        self.reqHistoricalData(
            reqId,
            contract,
            "",            # endDateTime ("" = now)
            duration,      # duration parameter
            bar_size,      # bar size parameter
            "TRADES",      # data type
            0,             # useRTH
            1,             # formatDate
            False,         # keepUpToDate
            []
        )

        for _ in range(30):
            time.sleep(0.5)
            with self.lock:
                if reqId in self.historical_data:
                    df = pd.DataFrame(self.historical_data[reqId])
                    df["date"] = pd.to_datetime(df["date"])
                    return df
        return None

    def place_market_order_with_id(self, symbol, quantity, action, order_id):
        """Place market order with specific order ID"""
        # Ensure quantity is an integer to avoid fractional share errors
        quantity = int(quantity)
        print(f"Market order: {symbol} {action} {quantity} shares")
        
        contract = self.stock_contract(symbol)
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        self.placeOrder(order_id, contract, order)
        
        # Wait for fill
        for _ in range(60):
            time.sleep(1)
            with self.lock:
                if order_id in self.order_status:
                    status = self.order_status[order_id].get('status', '')
                    if status == 'Filled':
                        fill_price = self.order_status[order_id].get('avgFillPrice', 0)
                        return order_id, fill_price
                    elif status in ['Cancelled', 'Rejected']:
                        return order_id, -1
        
        return order_id, -1

    def place_limit_order_with_id(self, symbol, quantity, limit_price, action, order_id):
        """Place limit order with specific order ID"""
        # Ensure quantity is an integer to avoid fractional share errors
        quantity = int(quantity)
        print(f"Limit order: {symbol} {action} {quantity} shares at ${limit_price:.2f}")
        
        contract = self.stock_contract(symbol)
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        self.placeOrder(order_id, contract, order)
        
        # Wait for fill
        for _ in range(60):
            time.sleep(1)
            with self.lock:
                if order_id in self.order_status:
                    status = self.order_status[order_id].get('status', '')
                    if status == 'Filled':
                        fill_price = self.order_status[order_id].get('avgFillPrice', 0)
                        return order_id, fill_price
                    elif status in ['Cancelled', 'Rejected']:
                        return order_id, -1
        
        return order_id, -1

    def cancel_order(self, order_id):
        """Cancel an order by order ID"""
        self.cancelOrder(order_id)
        print(f"Cancelled order ID: {order_id}")

    def get_all_used_order_ids(self):
        """Get all order IDs that have been used (active and completed)"""
        with self.lock:
            return list(self.order_status.keys())

    def request_all_open_orders(self):
        """Request all open orders from IBKR"""
        self.reqAllOpenOrders()
        time.sleep(2)  # Wait for response
        return list(self.order_status.keys())

    def get_next_order_id_from_ibkr(self):
        """Get the next available order ID directly from IBKR"""
        with self.lock:
            if self.next_valid_id is not None:
                next_id = self.next_valid_id
                self.next_valid_id += 1  # Increment for next use
                return next_id
            return None
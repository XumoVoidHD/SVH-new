import threading
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import pandas as pd
from typing import Optional, Dict, Callable, List, Generator, Tuple
import pytz

# Replace Angel WebSocket with Schwab wrapper
from .schwab_broker import SchwabBroker
from db.sqldb import SQLDB

class OrderStatus(Enum):
    NEW = 'NEW'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'
    FILLED = 'FILLED'
    CANCELED = 'CANCELED'

class PositionStatus(Enum):
    OPEN = 'OPEN'
    CLOSED = 'CLOSED'

@dataclass
class Order:
    id: int
    symbol: str
    qty: float
    order_type: str
    price: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    triggerprice: Optional[float]
    status: OrderStatus = OrderStatus.NEW
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.timezone('America/Chicago')))
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None

@dataclass
class Trade:
    order_id: int
    symbol: str
    qty: float
    exec_price: float
    commission: float
    pnl: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.timezone('America/Chicago')))

@dataclass
class Position:
    symbol: str
    qty: float = 0.0  # Always positive
    direction: int = 0  # 1 for long, -1 for short, 0 for no position
    open_qty: float = 0.0  # Total quantity opened
    close_qty: float = 0.0  # Total quantity closed
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    status: str = "CLOSED"
    open_time: datetime = None
    close_time: datetime = None
    open_price: float = 0.0
    close_price: float = 0.0
    trade_ids: List[str] = field(default_factory=list)
    order_ids: List[str] = field(default_factory=list)

    def update_unrealized(self, current_price: float) -> None:
        """Update unrealized PnL based on current price."""
        if self.qty == 0:
            self.unrealized_pnl = 0
            print(f"{self.symbol} is closed")
            return
        if self.direction == 1:  # Long position
            self.unrealized_pnl = (current_price - self.avg_price) * self.qty
        else:  # Short position
            self.unrealized_pnl = (self.avg_price - current_price) * self.qty

    def can_update(self, new_direction: int, new_qty: float) -> bool:
        """Check if position can be updated with new direction and quantity."""
        if self.status == "CLOSED":
            return True
        if self.direction == 0:
            return True
        if self.direction != new_direction:
            return False
        return True

    def update(self, qty: float, price: float, direction: int, trade_id: str, order_id: str) -> None:
        """Update position with new trade."""
        if not self.can_update(direction, qty):
            raise ValueError(f"Cannot update position with different direction: {direction}")

        if self.status == "CLOSED":
            self.status = "OPEN"
            self.direction = direction
            self.open_time = datetime.now(pytz.timezone('America/Chicago'))
            self.open_price = price

        self.qty += qty
        self.open_qty += qty
        self.avg_price = ((self.avg_price * (self.open_qty - qty)) + (price * qty)) / self.open_qty
        self.trade_ids.append(trade_id)
        self.order_ids.append(order_id)

    def close(self, qty: float, price: float, trade_id: str, order_id: str) -> float:
        """Close part or all of the position."""
        if qty > self.qty:
            raise ValueError(f"Cannot close more than current position: {qty} > {self.qty}")

        pnl = 0
        if self.direction == 1:  # Long position
            pnl = (price - self.avg_price) * qty
        else:  # Short position
            pnl = (self.avg_price - price) * qty

        self.qty -= qty
        self.close_qty += qty
        self.realized_pnl += pnl
        self.trade_ids.append(trade_id)
        self.order_ids.append(order_id)

        if self.qty == 0:
            self.status = "CLOSED"
            self.close_time = datetime.now(pytz.timezone('America/Chicago'))
            self.close_price = price
            self.direction = 0

        return pnl

class ForwardBroker:
    """
    MVP broker: MARKET, LIMIT, STOP-MARKET, STOP-LIMIT, SL/TP, basic PnL.
    """

    SUPPORTED_ORDER_TYPES = {'MARKET', 'LIMIT', 'STOP-MARKET', 'STOP-LIMIT'}

    def __init__(
        self,
        initial_balance: float,
        spread: float = 0.0,
        commission_fixed: float = 0.0,
        commission_rel: float = 0.0,
        broker=None
    ):
        self.broker = broker
        self.cash = initial_balance
        self.spread = spread
        self.commission = 1
        self.commission_fixed = commission_fixed
        self.commission_rel = commission_rel
        self.positions: Dict[str, Position] = {}
       
        self.trades: List[Trade] = []
        self.market_data: Dict[str, float] = {}
        self._order_id = 1
        self._lock = threading.RLock()
        self.on_order_filled: List[Callable[[Order], None]] = []
        self.on_trade: List[Callable[[Trade], None]] = []
        
        # Start with empty symbols list - only track symbols with orders
        self.symbols = []
        
        # Initialize Schwab broker for price updates
        self.schwab_broker = SchwabBroker()
        
        # Remove WebSocket dependencies - we'll use polling instead
        self._processing_thread = None
        self._stop_event = threading.Event()
        self.db = SQLDB()  # Initialize the SQLDB instance
        self.session_id = datetime.now(pytz.timezone('America/Chicago')).strftime("%Y%m%d_%H%M%S")  # Unique session ID
        self.orders: Dict[int, Order] = {}
        self.repopulate(self.db.get_orders(),self.db.get_trades(),self.get_positions())
        
        # Start order processing thread
        self._processing_thread = threading.Thread(target=self._process_orders_continuously, name="PNL Thread",daemon=False)
        self._processing_thread.daemon = True
        self._processing_thread.start()
    
    def _initialize_market_data(self):
        """Initialize market data for all symbols by getting current prices from Schwab."""
        print("Initializing market data from Schwab...")
        for symbol in self.symbols:
            try:
                current_price = self.schwab_broker.get_current_price(symbol)
                self.update_market_data(symbol, current_price)
                print(f"Initialized {symbol}: ${current_price:.2f}")
            except Exception as e:
                print(f"Error initializing market data for {symbol}: {e}")
        print("Market data initialization complete.")
    
    def add_symbol(self, symbol: str):
        """Add a new symbol to track and get its current price."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            try:
                current_price = self.schwab_broker.get_current_price(symbol)
                self.update_market_data(symbol, current_price)
                print(f"Added {symbol}: ${current_price:.2f}")
            except Exception as e:
                print(f"Error adding symbol {symbol}: {e}")
        else:
            print(f"Symbol {symbol} is already being tracked.")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from tracking."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.market_data:
                del self.market_data[symbol]
            print(f"Removed {symbol} from tracking.")
        else:
            print(f"Symbol {symbol} is not being tracked.")
    
    def cleanup_unused_symbols(self):
        """Remove symbols from tracking that no longer have orders or positions."""
        symbols_to_remove = []
        
        for symbol in self.symbols:
            # Check if symbol has any open orders
            has_orders = any(order.symbol == symbol and order.status == OrderStatus.NEW 
                           for order in self.orders.values())
            
            # Check if symbol has any open positions
            has_positions = symbol in self.positions and self.positions[symbol].qty != 0
            
            # If no orders and no positions, remove from tracking
            if not has_orders and not has_positions:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            self.remove_symbol(symbol)

    def repopulate(self,data_order: list,data_trades:list,data_pos:list):
        for order in data_order:
            order_id = order[1]
            symbol = order[2]
            qty = order[3]
            order_type = order[4]
            price = order[5]
            sl = order[6]
            tp = order[7]
            triggerprice = order[8]
            status = OrderStatus(order[9])
            timeStamp = order[10]
            filled_qty = order[11]
            fill_price = order[12]
            fill_time = order[13] 
            self.new_order(order_id=order_id,
                            symbol=symbol,
                            qty=qty,
                            order_type=order_type,
                            price=price,
                            sl = sl,
                            tp=tp,
                            triggerprice=triggerprice,
                            status = status,
                            timeStamp=timeStamp,
                            fill_price=fill_price,
                            fill_qty=  filled_qty,
                            fill_time=fill_time,
                            is_repopulate=True
            )
        
        for d in data_trades:
            order_id = d[1]
            symbol = d[2]
            qty = d[3]
            exec_price = d[4]
            commision = d[5]
            pnl = d[6]
            timeStamp = d[7]
            self.trades.append(Trade(
                order_id=order_id,
                symbol=symbol,
                qty=qty,
                exec_price=exec_price,
                commission=commision,
                pnl=pnl,
                timestamp=timeStamp
            ))

        for d in data_pos:
            symbol = d[2]
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=d[3],
                direction=d[4],
                open_qty=d[5],
                close_qty=d[6],
                avg_price=d[7],
                unrealized_pnl=d[8],
                realized_pnl=d[9],
                sl=d[10],
                tp=d[11],
                status=d[12],
                open_time=d[13],
                close_time=d[14],
                open_price=d[15],
                close_price=d[16],
                trade_ids=d[17].split(',') if d[17] else [],
                order_ids=d[18].split(',') if d[18] else []
            )

        # Add symbols from existing orders and positions to tracking list
        symbols_to_track = set()
        
        # Add symbols from orders
        for order in self.orders.values():
            symbols_to_track.add(order.symbol)
        
        # Add symbols from positions
        for position in self.positions.values():
            symbols_to_track.add(position.symbol)
        
        # Add symbols to tracking list and get initial prices
        for symbol in symbols_to_track:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                try:
                    current_price = self.schwab_broker.get_current_price(symbol)
                    self.update_market_data(symbol, current_price)
                    print(f"Added {symbol} to tracking from repopulate: ${current_price:.2f}")
                except Exception as e:
                    print(f"Error getting initial price for {symbol} during repopulate: {e}")
        
        print(f" data = {len(self.orders.values())} {len(self.trades)} {len(self.positions.values())}")
        print(f" tracking symbols: {self.symbols}")

    @property
    def equity(self) -> float:
        with self._lock:
            unreal = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized = sum(pos.realized_pnl for pos in self.positions.values())
            print(f"Cash: {self.cash}, Realized: {realized}, Unrealized: {unreal}")
            return self.cash + realized + unreal

    @property
    def margin_used(self) -> float:
        with self._lock:
            print(f"Margin: {sum(abs(pos.qty) * pos.avg_price for pos in self.positions.values())}")
            return sum(abs(pos.qty) * pos.avg_price for pos in self.positions.values())

    @property
    def free_margin(self) -> float:
        print(f"Equity: {self.equity}, Margin: {self.margin_used}")
        return self.equity - self.margin_used

    def _validate_order(self, order: Order):
        if order.order_type not in self.SUPPORTED_ORDER_TYPES:
            raise ValueError(f"Unsupported order type {order.order_type}")
        if order.qty == 0:
            raise ValueError("Quantity must be non-zero")
        if order.order_type == 'LIMIT' and order.price is None:
            raise ValueError("Limit orders require a price")
        if order.order_type == 'STOP-LIMIT' and (order.price is None or order.triggerprice is None):
            raise ValueError("Stop-limit orders require price and triggerprice")
            
        # Get current position for the symbol
        current_pos = self.positions.get(order.symbol)
        
        # For market orders, always use the market data price from websocket
        if order.order_type == 'MARKET':
            market_price = self.market_data.get(order.symbol)
            
            # Price used for calculations in pyramiding
            self.market_close = market_price
            if not market_price:
                raise ValueError(f"No market data available for {order.symbol}")
            notional = abs(order.qty) * market_price
        else:
            notional = abs(order.qty) * (order.price if order.price is not None else self.market_data.get(order.symbol, 0))
        
        #Vedansh
        # Check if order would reverse position direction
        if current_pos and current_pos.qty > 0:
            if (current_pos.direction > 0 and order.qty < 0) or (current_pos.direction < 0 and order.qty > 0):
                if abs(order.qty) > current_pos.qty:
                    raise ValueError("Order quantity exceeds current position size")
            
        # Margin check for market and limit orders
        # Vedansh 
        if order.order_type in {'MARKET', 'LIMIT'} and notional > self.free_margin:
            print(f"Insufficient margin. Required: {notional}, Available: {self.free_margin}")
        #    raise ValueError(f"Insufficient margin. Required: {notional}, Available: {self.free_margin}")
        
        

    def new_order(
        self,

        symbol: str,
        qty: float,
        order_id= None,
        order_type: str = 'MARKET',
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        triggerprice: Optional[float] = None,
        timeStamp = None,
        fill_qty = 0,
        fill_price= None,
        fill_time = None,
        status = OrderStatus.NEW,
        is_repopulate = False
    ) -> int:
        with self._lock:
            oid = self._order_id
            if order_id:
                oid = self._order_id
            order = Order(
                id=oid,
                symbol=symbol,
                qty=qty,
                order_type=order_type,
                price=price,
                sl=sl,
                tp=tp,
                status=status,
                triggerprice=triggerprice,
                timestamp=timeStamp,
                fill_price=fill_price,
                filled_qty=fill_qty,
                fill_time=fill_time
                
            )
            if not is_repopulate:
                self._validate_order(order)
                
                # Add symbol to tracking list if not already there
                if order.symbol not in self.symbols:
                    self.symbols.append(order.symbol)
                    # Get initial price for the symbol
                    try:
                        current_price = self.schwab_broker.get_current_price(order.symbol)
                        self.update_market_data(order.symbol, current_price)
                        print(f"Added {order.symbol} to tracking: ${current_price:.2f}")
                    except Exception as e:
                        print(f"Error getting initial price for {order.symbol}: {e}")
                
            self._order_id += 1
            self.orders[oid] = order

            
            if not is_repopulate:
                # Save order to database
                self.db.insert_order(
                    order_id=order.id,
                    symbol=order.symbol,
                    qty=order.qty,
                    order_type=order.order_type,
                    price=order.price,
                    sl=order.sl,
                    tp=order.tp,
                    triggerprice=order.triggerprice,
                    status=str(order.status.value),
                    filled_qty=order.filled_qty,
                    fill_price=order.fill_price,
                    fill_time=order.fill_time
                )
           
            return oid

    def cancel_order(self, order_id: int):
        with self._lock:
            order = self.orders.get(order_id)
            if order and order.status == OrderStatus.NEW:
                order.status = OrderStatus.CANCELED
                order.fill_time = datetime.now(pytz.timezone('America/Chicago'))
                
                # Update order in database
                self.db.update_order(
                    order_id=order.id,
                    status=order.status.value,
                    fill_time=order.fill_time
                )

    def get_open_orders(self) -> List[Order]:
        with self._lock:
            return [o for o in self.orders.values() if o.status == OrderStatus.NEW]

    def get_closed_orders(self) -> List[Order]:
        with self._lock:
            return [o for o in self.orders.values() if o.status in {OrderStatus.FILLED, OrderStatus.CANCELED}]

    def get_all_trades(self) -> List[Trade]:
        return list(self.trades)

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            return {sym: pos for sym, pos in self.positions.items()}

    def update_market_data(self, symbol: str, price: float):
        with self._lock:
            # Schwab prices are already in correct format, no need to divide by 100
            self.market_data[symbol] = price
    def _process_orders(self, symbol: str):
        price = self.market_data[symbol]
        for order in list(self.orders.values()):
            if (order.status != "NEW" and order.status != OrderStatus.NEW) or order.symbol != symbol:
                # print("order status",order.status,"order symbol",order.symbol,"symbol",symbol)
                continue
            filled = False; exec_price = price
            if order.order_type == 'MARKET':
                exec_price *= (1 + self.spread if order.qty > 0 else 1 - self.spread)
                filled = True
            elif order.order_type == 'LIMIT':
                if (order.qty > 0 and price <= order.price) or (order.qty < 0 and price >= order.price):
                    exec_price = order.price; filled = True
            elif order.order_type == 'STOP-MARKET':
                if (order.qty > 0 and price >= order.triggerprice) or (order.qty < 0 and price <= order.triggerprice):
                    exec_price *= (1 + self.spread if order.qty > 0 else 1 - self.spread); filled = True
            elif order.order_type == 'STOP-LIMIT':
                if order.filled == False and order.fill_price is None:
                    if (order.qty > 0 and price >= order.triggerprice) or (order.qty < 0 and price <= order.triggerprice):
                        # trigger stage
                        if (order.qty > 0 and price <= order.price) or (order.qty < 0 and price >= order.price):
                            exec_price = order.price; filled = True
            if filled:
                self._fill_order(order, exec_price)
                
    def is_closing_order(self, order, position):
        # If there is no position, it can't be closing
        if not position or position.qty == 0:
            return False
        # Check if the order is in the opposite direction of the position
        if (position.direction == 1 and order.qty < 0) or (position.direction == -1 and order.qty > 0):
            # Quantity to close should not exceed current position
            if abs(order.qty) <= abs(position.qty):
                return True
        return False


    def _fill_order(self, order: Order, price: float) -> None:
        """Fill an order at the given price."""
        try:
            required_cash = 0
            # Check if we have enough cash
            position = self.positions.get(order.symbol)
            if self.is_closing_order(order, position):
                print(f"Closing Position: {order.symbol}")
            else:
                # Opening or adding; check for cash
                required_cash = abs(order.qty * price)
                if self.cash < required_cash:
                    raise ValueError(f"Insufficient cash: {self.cash} < {required_cash}")


            # Calculate commission
            # Vedansh
            # commission = abs(order.qty * price * self.commission)
            commission = 0

            # Create trade record
            trade_id = f"TRADE_{len(self.trades) + 1}"
            trade = Trade(
                order_id=order.id,
                symbol=order.symbol,
                qty=order.qty,
                exec_price=price,
                commission=commission,
                pnl=0.0,
                timestamp=datetime.now(pytz.timezone('America/Chicago'))
            )
            self.trades.append(trade)

            # Update position
            position = self.positions.get(order.symbol)
            direction = 1 if order.qty > 0 else -1
            qty = abs(order.qty)
            if position is None or position.status == "CLOSED":
                # Create new position
                position = Position(
                    symbol=order.symbol,
                    qty=qty,
                    direction=direction,
                    open_qty=qty,
                    close_qty=0.0,
                    avg_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    status="OPEN",
                    open_time=datetime.now(pytz.timezone('America/Chicago')),
                    open_price=price,
                    trade_ids=[trade_id],
                    order_ids=[str(order.id)]
                )
                self.positions[order.symbol] = position
                self.db.insert_position(
                    symbol=position.symbol,
                    qty=position.qty,
                    direction=position.direction,
                    open_qty=position.open_qty,
                    close_qty=position.close_qty,
                    avg_price=position.avg_price,
                    unrealized_pnl=position.unrealized_pnl,
                    realized_pnl=position.realized_pnl,
                    sl=position.sl,
                    tp=position.tp,
                    status=position.status,
                    open_time=position.open_time,
                    close_time=position.close_time,
                    open_price=position.open_price,
                    close_price=position.close_price,
                    trade_ids=",".join(map(str, position.trade_ids)),
                    order_ids=",".join(map(str, position.order_ids)),
                    session_id=self.session_id
                )
            else:
                # Update existing position
                if direction == position.direction:
                    # Adding to position
                    position.update(qty, price, direction, trade_id, str(order.id))
                else:
                    # Closing position
                    pnl = position.close(qty, price, trade_id, str(order.id))
                    trade.pnl = pnl
                self.db.update_position(
                    position_id=position.symbol,
                    qty=position.qty,
                    direction=position.direction,
                    open_qty=position.open_qty,
                    close_qty=position.close_qty,
                    avg_price=position.avg_price,
                    unrealized_pnl=position.unrealized_pnl,
                    realized_pnl=position.realized_pnl,
                    sl=position.sl,
                    tp=position.tp,
                    status=position.status,
                    open_time=position.open_time,
                    close_time=position.close_time,
                    open_price=position.open_price,
                    close_price=position.close_price,
                    trade_ids=",".join(map(str, position.trade_ids)),
                    order_ids=",".join(map(str, position.order_ids)),
                    session_id=self.session_id
                )

            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_qty = order.qty
            order.fill_price = price
            order.fill_time = datetime.now(pytz.timezone('America/Chicago'))

            # Update cash based on order direction
            if order.qty > 0:  # Buy order - deduct cash
                self.cash -= (required_cash + commission)
            else:  # Sell order - add cash back
                self.cash += (abs(order.qty * price) - commission)

            # Update database
            self.db.insert_trade(
                order_id=order.id,
                symbol=order.symbol,
                qty=order.qty,
                exec_price=price,
                commission=commission,
                pnl=trade.pnl,
                timestamp=trade.timestamp
            )

            self.db.update_order(
                order_id=order.id,
                status=str(order.status.value),
                filled_qty=order.filled_qty,
                fill_price=order.fill_price,
                fill_time=order.fill_time
            )

        except Exception as e:
            print(f"Error filling order {order}: {str(e)}")
            raise

    def _check_sl_tp(self, symbol: str):
        price = self.market_data[symbol]
        pos = self.positions.get(symbol)
        if pos is None or pos.qty == 0:
            return

        if pos.sl != 0:
            if pos.sl is not None and ((pos.direction > 0 and price <= pos.sl) or (pos.direction < 0 and price >= pos.sl)):
                # Place close order but don't mark as closed yet
                self.new_order(
                symbol=symbol,
                qty=-pos.qty,
                order_type='MARKET'
            )

            pos.sl = None

        if pos.tp != 0:
            if pos.tp is not None and ((pos.direction > 0 and price >= pos.tp) or (pos.direction < 0 and price <= pos.tp)):
                # Place close order but don't mark as closed yet
                self.new_order(
                    symbol=symbol,
                    qty=-pos.qty,
                    order_type='MARKET'
                )

                pos.tp = None

    def get_balance(self):
        return self.cash
    def place_order(self, order_params):
        symbol = order_params.get('symbol', order_params.get('tradingsymbol', ''))
        
        # Get current price from Schwab for the symbol
        try:
            current_price = self.schwab_broker.get_current_price(symbol)
            self.update_market_data(symbol, current_price)
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None, None
        
        # Convert Schwab order parameters
        if order_params.get("side", order_params.get("transactiontype", "")) == "SELL":
            qty = -float(order_params.get('quantity', order_params.get('qty', 1)))
        else:
            qty = float(order_params.get('quantity', order_params.get('qty', 1)))
        
        # Map order types to Schwab format
        order_type = str(order_params.get('ordertype', 'MARKET')).upper()
        if order_type not in self.SUPPORTED_ORDER_TYPES:
            order_type = 'MARKET'  # Default to market order
        
        oid = self.new_order(
            symbol=symbol,
            qty=qty,
            order_type=order_type,
            price=float(order_params.get('price', 0)),
            sl=float(order_params.get('stoploss', 0)),
            tp=float(order_params.get('take_profit', order_params.get('squareoff', 0))),
            triggerprice=float(order_params.get('triggerprice', 0))
        )

        if order_type == "MARKET":
            return oid, current_price
        else:
            return oid, None
    def broker_summary(self):
        print(f"Cash: {self.cash}, Margin Used: {self.margin_used}, Free Margin: {self.free_margin}")
        print(f"Positions: {self.positions}")
        print(f"Orders: {self.orders}")
        print(f"Trades: {self.trades}")
    # Convenience methods
    def buy_market(self, symbol: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None):
        return self.new_order(symbol=symbol, qty=qty, order_type='MARKET', sl=sl, tp=tp)

    def sell_market(self, symbol: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None):
        return self.new_order(symbol=symbol, qty=-qty, order_type='MARKET', sl=sl, tp=tp)

    def buy_limit(self, symbol: str, qty: float, price: float, sl: Optional[float] = None, tp: Optional[float] = None):
        return self.new_order(symbol=symbol, qty=qty, order_type='LIMIT', price=price, sl=sl, tp=tp)

    def sell_limit(self, symbol: str, qty: float, price: float, sl: Optional[float] = None, tp: Optional[float] = None):
        return self.new_order(symbol=symbol, qty=-qty, order_type='LIMIT', price=price, sl=sl, tp=tp)

    # Schwab-specific convenience methods
    def schwab_buy_market(self, symbol: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None):
        """Place a market buy order using Schwab parameters."""
        order_params = {
            'symbol': symbol,
            'side': 'BUY',
            'quantity': qty,
            'ordertype': 'MARKET',
            'stoploss': sl or 0,
            'take_profit': tp or 0
        }
        return self.place_order(order_params)
    
    def schwab_sell_market(self, symbol: str, qty: float, sl: Optional[float] = None, tp: Optional[float] = None):
        """Place a market sell order using Schwab parameters."""
        order_params = {
            'symbol': symbol,
            'side': 'SELL',
            'quantity': qty,
            'ordertype': 'MARKET',
            'stoploss': sl or 0,
            'take_profit': tp or 0
        }
        return self.place_order(order_params)
    
    def schwab_buy_limit(self, symbol: str, qty: float, price: float, sl: Optional[float] = None, tp: Optional[float] = None):
        """Place a limit buy order using Schwab parameters."""
        order_params = {
            'symbol': symbol,
            'side': 'BUY',
            'quantity': qty,
            'ordertype': 'LIMIT',
            'price': price,
            'stoploss': sl or 0,
            'take_profit': tp or 0
        }
        return self.place_order(order_params)
    
    def schwab_sell_limit(self, symbol: str, qty: float, price: float, sl: Optional[float] = None, tp: Optional[float] = None):
        """Place a limit sell order using Schwab parameters."""
        order_params = {
            'symbol': symbol,
            'side': 'SELL',
            'quantity': qty,
            'ordertype': 'LIMIT',
            'price': price,
            'stoploss': sl or 0,
            'take_profit': tp or 0
        }
        return self.place_order(order_params)


    def _process_orders_continuously(self):
        """Continuously process orders for all symbols by polling Schwab for prices."""
        cleanup_counter = 0
        while not self._stop_event.is_set():
            try:
                # Poll for current prices from Schwab for all symbols
                for symbol in self.symbols:
                    try:
                        # Get current price from Schwab
                        current_price = self.schwab_broker.get_current_price(symbol)
                        self.update_market_data(symbol, current_price)
                        
                        # Always update unrealized PnL for all positions
                        if symbol in self.positions:
                            pos = self.positions[symbol]
                            if pos.qty != 0:  # Only update for open positions
                                pos.update_unrealized(current_price)
                                # Update position in database with new unrealized PnL
                                self.db.update_position(
                                    position_id=str(pos.symbol),
                                    unrealized_pnl=pos.unrealized_pnl
                                )
                            else:
                                self.db.update_closed_position(
                                    position_id=str(pos.symbol),
                                    unrealized_pnl=0
                                )
                        
                        # Process orders and check SL/TP
                        self._process_orders(symbol)
                        self._check_sl_tp(symbol)
                    except Exception as e:
                        print(f"Error processing orders for symbol {symbol}: {e}")
                
                # Clean up unused symbols every 60 iterations (5 minutes)
                cleanup_counter += 1
                if cleanup_counter >= 60:
                    self.cleanup_unused_symbols()
                    cleanup_counter = 0
                        
            except Exception as e:
                print(f"Error in order processing thread: {e}")
            
            # Sleep for a longer duration since we're polling instead of using WebSocket
            time.sleep(5)  # Poll every 5 seconds to avoid rate limiting
        
    def set_pnl_to_zero(self, symbol):
        self.db.update_closed_position(
            position_id=symbol,
            unrealized_pnl=0
        )
        
    def total_pnl(self) -> float:
        return self.db.total_realized_pnl() + self.db.total_unrealized_pnl()
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        return self.db.get_unrealized_pnl_by_symbol(symbol)
    
    def test(self):
        self.db.update_closed_position(
            position_id="BANKNIFTY26JUN2556300CE",
            unrealized_pnl=0
        )

    def filled_check(self, symbol):
        """
        Check if all orders for a symbol are filled.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            bool: True if all orders for the symbol are filled, False if any are still NEW
        """
        try:
            # Get all orders for this symbol from database
            orders = self.db.get_orders_by_symbol(symbol)
            
            if not orders:
                # No orders found for this symbol
                return True
            
            # Check if any order has status 'NEW'
            for order in orders:
                if order[9] == 'NEW':  # Assuming status is at index 9 based on the repopulate method
                    return False
            
            # All orders are filled
            return True
            
        except Exception as e:
            print(f"Error checking status for {symbol}: {e}")
            return False

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const Joi = require('joi');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Validate required environment variables
if (!process.env.MONGODB_URI) {
  console.error('ERROR: MONGODB_URI environment variable is required');
  process.exit(1);
}

// Security middleware
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// CORS configuration
const corsOptions = {
  origin: process.env.CORS_ORIGIN || '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Device-ID'],
  credentials: true
};
app.use(cors(corsOptions));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// MongoDB connection with retry logic
const connectToMongoDB = async () => {
  try {
    await mongoose.connect(process.env.MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000,
    });
    console.log('Connected to MongoDB successfully');
  } catch (error) {
    console.error('MongoDB connection failed:', error.message);
    console.log('Retrying connection in 5 seconds...');
    setTimeout(connectToMongoDB, 5000);
  }
};

connectToMongoDB();

// MongoDB event handlers
mongoose.connection.on('connected', () => {
  console.log('Mongoose connected to MongoDB');
});

mongoose.connection.on('error', (err) => {
  console.error('Mongoose connection error:', err);
});

mongoose.connection.on('disconnected', () => {
  console.log('Mongoose disconnected');
});

// Sensor Reading Schema - Updated to match Flutter field names
const sensorReadingSchema = new mongoose.Schema({
  sensorId: { type: String, required: true, index: true },
  gasLevel: { type: Number, required: true, min: 0 },
  unit: { type: String, default: '%LEL', required: true },
  location: String,
  temperature: Number,
  humidity: Number,
  pressure: Number,
  latitude: Number,
  longitude: Number,
  status: { 
    type: String, 
    enum: ['online', 'offline', 'maintenance', 'active', 'inactive'], 
    default: 'online' 
  },
  type: { 
    type: String, 
    enum: ['gas', 'temperature', 'humidity', 'pressure', 'gasDetector'], 
    default: 'gas' 
  },
  isCalibrated: { type: Boolean, default: false },
  calibrationOffset: Number,
  timestamp: { type: Date, default: Date.now, required: true },
  deviceId: String,
  batchId: String,
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  syncedAt: { type: Date, default: Date.now },
  syncStatus: {
    type: String,
    enum: ['pending', 'synced', 'failed'],
    default: 'pending'
  },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// Alert Schema
const alertSchema = new mongoose.Schema({
  sensorId: { type: String, required: true, index: true },
  alertType: { 
    type: String, 
    enum: ['gas_high', 'gas_critical', 'sensor_offline', 'calibration_due'], 
    required: true 
  },
  severity: { 
    type: String, 
    enum: ['low', 'medium', 'high', 'critical'], 
    required: true 
  },
  message: { type: String, required: true },
  value: Number,
  threshold: Number,
  isActive: { type: Boolean, default: true },
  acknowledgedAt: Date,
  acknowledgedBy: String,
  createdAt: { type: Date, default: Date.now },
  resolvedAt: Date,
  deviceId: String
});

// Sensor Configuration Schema
const sensorConfigSchema = new mongoose.Schema({
  sensorId: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  location: String,
  type: {
    type: String,
    enum: ['gasDetector', 'temperature', 'humidity', 'pressure'],
    default: 'gasDetector'
  },
  thresholds: {
    warning: { type: Number, default: 25 },
    critical: { type: Number, default: 50 },
    danger: { type: Number, default: 75 }
  },
  calibrationInterval: { type: Number, default: 30 }, // days
  lastCalibration: Date,
  nextCalibrationDue: Date,
  isActive: { type: Boolean, default: true },
  alertEnabled: { type: Boolean, default: true },
  reportingInterval: { type: Number, default: 60 }, // seconds
  description: String,
  settings: mongoose.Schema.Types.Mixed,
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// Add pre-save middleware to update timestamps
sensorReadingSchema.pre('save', function(next) {
  this.updatedAt = new Date();
  next();
});

sensorConfigSchema.pre('save', function(next) {
  this.updatedAt = new Date();
  next();
});

// Create models
const SensorReading = mongoose.model('SensorReading', sensorReadingSchema);
const Alert = mongoose.model('Alert', alertSchema);
const SensorConfig = mongoose.model('SensorConfig', sensorConfigSchema);

// Validation schemas
const sensorReadingValidation = Joi.object({
  sensorId: Joi.string().required(),
  gasLevel: Joi.number().min(0).required(),
  unit: Joi.string().default('%LEL'),
  location: Joi.string().optional(),
  temperature: Joi.number().optional(),
  humidity: Joi.number().optional(),
  pressure: Joi.number().optional(),
  latitude: Joi.number().optional(),
  longitude: Joi.number().optional(),
  status: Joi.string().valid('online', 'offline', 'maintenance', 'active', 'inactive').default('online'),
  type: Joi.string().valid('gas', 'temperature', 'humidity', 'pressure', 'gasDetector').default('gas'),
  isCalibrated: Joi.boolean().default(false),
  calibrationOffset: Joi.number().optional(),
  timestamp: Joi.date().default(() => new Date()),
  deviceId: Joi.string().optional(),
  metadata: Joi.object().default({})
});

// Utility function to create alerts
async function checkAndCreateAlerts(reading) {
  try {
    const config = await SensorConfig.findOne({ sensorId: reading.sensorId });
    if (!config || !config.alertEnabled) return;

    const alerts = [];
    
    if (reading.gasLevel >= config.thresholds.danger) {
      alerts.push({
        sensorId: reading.sensorId,
        alertType: 'gas_critical',
        severity: 'critical',
        message: `Critical gas level detected: ${reading.gasLevel}${reading.unit}`,
        value: reading.gasLevel,
        threshold: config.thresholds.danger,
        deviceId: reading.deviceId
      });
    } else if (reading.gasLevel >= config.thresholds.critical) {
      alerts.push({
        sensorId: reading.sensorId,
        alertType: 'gas_high',
        severity: 'high',
        message: `High gas level detected: ${reading.gasLevel}${reading.unit}`,
        value: reading.gasLevel,
        threshold: config.thresholds.critical,
        deviceId: reading.deviceId
      });
    }

    // Check if sensor is offline (no recent readings)
    const lastReading = await SensorReading.findOne(
      { sensorId: reading.sensorId },
      {},
      { sort: { timestamp: -1 } }
    );
    
    if (lastReading && Date.now() - lastReading.timestamp > 300000) { // 5 minutes
      alerts.push({
        sensorId: reading.sensorId,
        alertType: 'sensor_offline',
        severity: 'medium',
        message: `Sensor ${reading.sensorId} appears to be offline`,
        deviceId: reading.deviceId
      });
    }

    if (alerts.length > 0) {
      await Alert.insertMany(alerts);
      console.log(`Created ${alerts.length} alerts for sensor ${reading.sensorId}`);
    }
  } catch (error) {
    console.error('Error creating alerts:', error);
  }
}

// Middleware for request validation
const validateRequest = (schema) => {
  return (req, res, next) => {
    const { error } = schema.validate(req.body);
    if (error) {
      return res.status(400).json({
        success: false,
        error: 'Validation failed',
        details: error.details.map(detail => ({
          field: detail.path.join('.'),
          message: detail.message
        }))
      });
    }
    next();
  };
};

// Routes

// Health check with detailed status
app.get('/health', async (req, res) => {
  try {
    // Check database connection
    const dbStatus = mongoose.connection.readyState === 1 ? 'connected' : 'disconnected';
    
    // Get basic stats
    const [readingsCount, alertsCount] = await Promise.all([
      SensorReading.countDocuments(),
      Alert.countDocuments({ isActive: true })
    ]);

    res.json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      database: dbStatus,
      stats: {
        totalReadings: readingsCount,
        activeAlerts: alertsCount
      },
      environment: process.env.NODE_ENV || 'development'
    });
  } catch (error) {
    res.status(503).json({
      status: 'error',
      timestamp: new Date().toISOString(),
      error: 'Health check failed',
      details: error.message
    });
  }
});

// Sync sensor readings (batch upload)
app.post('/api/sensor-readings/sync', async (req, res) => {
  try {
    const { readings } = req.body;
    
    if (!readings || !Array.isArray(readings)) {
      return res.status(400).json({ 
        success: false,
        error: 'Invalid request format. Expected { readings: Array }' 
      });
    }

    if (readings.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No readings provided'
      });
    }

    if (readings.length > 1000) {
      return res.status(400).json({
        success: false,
        error: 'Batch size too large. Maximum 1000 readings per batch'
      });
    }

    const results = [];
    const batchId = new mongoose.Types.ObjectId().toString();
    const syncTimestamp = new Date();

    // Process each reading with validation
    for (const readingData of readings) {
      try {
        // Validate the reading data
        const { error, value: validatedData } = sensorReadingValidation.validate(readingData);
        
        if (error) {
          results.push({ 
            success: false, 
            error: `Validation error: ${error.details[0].message}`,
            sensorId: readingData.sensorId || 'unknown'
          });
          continue;
        }

        // Add batch metadata
        const readingToSave = {
          ...validatedData,
          batchId,
          syncedAt: syncTimestamp,
          // Ensure timestamp is a proper Date object
          timestamp: validatedData.timestamp instanceof Date 
            ? validatedData.timestamp 
            : new Date(validatedData.timestamp)
        };

        const reading = new SensorReading(readingToSave);
        const savedReading = await reading.save();
        
        // Check for alerts asynchronously
        setImmediate(() => checkAndCreateAlerts(savedReading));
        
        results.push({ 
          success: true, 
          id: savedReading._id,
          sensorId: savedReading.sensorId 
        });
      } catch (error) {
        console.error('Error saving reading:', error);
        results.push({ 
          success: false, 
          error: error.message,
          sensorId: readingData.sensorId || 'unknown'
        });
      }
    }

    const successCount = results.filter(r => r.success).length;
    const failureCount = results.length - successCount;
    
    res.json({
      success: failureCount === 0,
      batchId,
      totalProcessed: readings.length,
      successCount,
      failureCount,
      results,
      processedAt: syncTimestamp.toISOString()
    });
  } catch (error) {
    console.error('Batch sync error:', error);
    res.status(500).json({ 
      success: false,
      error: 'Internal server error during batch sync',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get sensor readings with enhanced filtering and pagination
app.get('/api/sensor-readings', async (req, res) => {
  try {
    const {
      sensorId,
      startDate,
      endDate,
      page = 1,
      limit = 50,
      sortBy = 'timestamp',
      sortOrder = 'desc',
      status,
      type
    } = req.query;

    // Validate pagination parameters
    const pageNum = Math.max(1, parseInt(page));
    const limitNum = Math.max(1, Math.min(100, parseInt(limit)));

    const query = {};
    
    if (sensorId) {
      query.sensorId = sensorId;
    }
    
    if (status) {
      query.status = status;
    }
    
    if (type) {
      query.type = type;
    }
    
    if (startDate || endDate) {
      query.timestamp = {};
      if (startDate) {
        const start = new Date(startDate);
        if (isNaN(start.getTime())) {
          return res.status(400).json({ error: 'Invalid startDate format' });
        }
        query.timestamp.$gte = start;
      }
      if (endDate) {
        const end = new Date(endDate);
        if (isNaN(end.getTime())) {
          return res.status(400).json({ error: 'Invalid endDate format' });
        }
        query.timestamp.$lte = end;
      }
    }

    const skip = (pageNum - 1) * limitNum;
    const validSortFields = ['timestamp', 'gasLevel', 'sensorId', 'createdAt'];
    const sortField = validSortFields.includes(sortBy) ? sortBy : 'timestamp';
    const sort = { [sortField]: sortOrder === 'desc' ? -1 : 1 };

    const [readings, totalCount] = await Promise.all([
      SensorReading.find(query)
        .sort(sort)
        .skip(skip)
        .limit(limitNum)
        .lean(),
      SensorReading.countDocuments(query)
    ]);

    res.json({
      success: true,
      data: readings,
      pagination: {
        currentPage: pageNum,
        totalPages: Math.ceil(totalCount / limitNum),
        totalCount,
        hasNextPage: skip + readings.length < totalCount,
        hasPrevPage: pageNum > 1,
        limit: limitNum
      },
      query: {
        sensorId,
        startDate,
        endDate,
        status,
        type
      }
    });
  } catch (error) {
    console.error('Error fetching readings:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch sensor readings',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get latest readings for all sensors
app.get('/api/sensor-readings/latest', async (req, res) => {
  try {
    const latestReadings = await SensorReading.aggregate([
      {
        $sort: { sensorId: 1, timestamp: -1 }
      },
      {
        $group: {
          _id: '$sensorId',
          latestReading: { $first: '$$ROOT' }
        }
      },
      {
        $replaceRoot: { newRoot: '$latestReading' }
      },
      {
        $sort: { timestamp: -1 }
      }
    ]);

    res.json({
      success: true,
      data: latestReadings,
      count: latestReadings.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching latest readings:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch latest readings',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Sync sensor configurations
app.post('/api/sensor-configs/sync', async (req, res) => {
  try {
    const { configs } = req.body;
    
    if (!configs || !Array.isArray(configs)) {
      return res.status(400).json({ 
        success: false,
        error: 'Invalid request format. Expected { configs: Array }' 
      });
    }

    if (configs.length > 100) {
      return res.status(400).json({
        success: false,
        error: 'Batch size too large. Maximum 100 configurations per batch'
      });
    }

    const results = [];
    const syncTimestamp = new Date();

    for (const configData of configs) {
      try {
        // Ensure required fields
        if (!configData.sensorId) {
          results.push({ 
            success: false, 
            error: 'sensorId is required',
            sensorId: 'unknown'
          });
          continue;
        }

        configData.updatedAt = syncTimestamp;
        
        const config = await SensorConfig.findOneAndUpdate(
          { sensorId: configData.sensorId },
          configData,
          { upsert: true, new: true, runValidators: true }
        );
        
        results.push({ 
          success: true, 
          id: config._id,
          sensorId: config.sensorId 
        });
      } catch (error) {
        console.error('Error syncing config:', error);
        results.push({ 
          success: false, 
          error: error.message,
          sensorId: configData.sensorId || 'unknown'
        });
      }
    }

    const successCount = results.filter(r => r.success).length;
    const failureCount = results.length - successCount;
    
    res.json({
      success: failureCount === 0,
      totalProcessed: configs.length,
      successCount,
      failureCount,
      results,
      processedAt: syncTimestamp.toISOString()
    });
  } catch (error) {
    console.error('Config sync error:', error);
    res.status(500).json({ 
      success: false,
      error: 'Internal server error during config sync',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get all sensor configurations
app.get('/api/sensor-configs', async (req, res) => {
  try {
    const { isActive } = req.query;
    
    const query = {};
    if (isActive !== undefined) {
      query.isActive = isActive === 'true';
    }

    const configs = await SensorConfig.find(query)
      .sort({ createdAt: -1 })
      .lean();

    res.json({
      success: true,
      data: configs,
      count: configs.length
    });
  } catch (error) {
    console.error('Error fetching configs:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch sensor configurations',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get active alerts
app.get('/api/alerts/active', async (req, res) => {
  try {
    const { severity, sensorId, limit = 100 } = req.query;
    
    const query = { isActive: true };
    
    if (severity) {
      query.severity = severity;
    }
    
    if (sensorId) {
      query.sensorId = sensorId;
    }

    const alerts = await Alert.find(query)
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .lean();

    res.json({
      success: true,
      data: alerts,
      count: alerts.length
    });
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch active alerts',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Acknowledge alert - FIXED: Now expects camelCase acknowledgedBy
app.put('/api/alerts/:id/acknowledge', async (req, res) => {
  try {
    const { id } = req.params;
    const { acknowledgedBy } = req.body; // Fixed: camelCase as expected

    if (!acknowledgedBy) {
      return res.status(400).json({
        success: false,
        error: 'acknowledgedBy field is required'
      });
    }

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid alert ID format'
      });
    }

    const alert = await Alert.findByIdAndUpdate(
      id,
      {
        acknowledgedAt: new Date(),
        acknowledgedBy,
        isActive: false
      },
      { new: true, runValidators: true }
    );

    if (!alert) {
      return res.status(404).json({ 
        success: false,
        error: 'Alert not found' 
      });
    }

    res.json({
      success: true,
      data: alert
    });
  } catch (error) {
    console.error('Error acknowledging alert:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to acknowledge alert',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get dashboard statistics
app.get('/api/dashboard/stats', async (req, res) => {
  try {
    const [
      totalReadings,
      activeAlerts,
      sensorCount,
      recentReadingsCount,
      criticalAlertsCount,
      avgGasLevels
    ] = await Promise.all([
      SensorReading.countDocuments(),
      Alert.countDocuments({ isActive: true }),
      SensorConfig.countDocuments({ isActive: true }),
      SensorReading.countDocuments({
        timestamp: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
      }),
      Alert.countDocuments({ 
        isActive: true, 
        severity: { $in: ['high', 'critical'] }
      }),
      // Get average gas levels by sensor for last 24 hours
      SensorReading.aggregate([
        {
          $match: {
            timestamp: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) }
          }
        },
        {
          $group: {
            _id: '$sensorId',
            avgGasLevel: { $avg: '$gasLevel' },
            maxGasLevel: { $max: '$gasLevel' },
            minGasLevel: { $min: '$gasLevel' },
            readingCount: { $sum: 1 },
            lastReading: { $max: '$timestamp' }
          }
        }
      ])
    ]);

    res.json({
      success: true,
      data: {
        totalReadings,
        activeAlerts,
        sensorCount,
        recentReadingsCount,
        criticalAlertsCount,
        avgGasLevels,
        systemHealth: {
          status: criticalAlertsCount > 0 ? 'warning' : 'healthy',
          uptime: process.uptime(),
          memoryUsage: process.memoryUsage()
        },
        lastUpdated: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error fetching dashboard stats:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch dashboard statistics',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Get alerts by sensor ID
app.get('/api/alerts/sensor/:sensorId', async (req, res) => {
  try {
    const { sensorId } = req.params;
    const { isActive, limit = 50 } = req.query;
    
    const query = { sensorId };
    if (isActive !== undefined) {
      query.isActive = isActive === 'true';
    }

    const alerts = await Alert.find(query)
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .lean();

    res.json({
      success: true,
      data: alerts,
      sensorId,
      count: alerts.length
    });
  } catch (error) {
    console.error('Error fetching sensor alerts:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch sensor alerts',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Bulk acknowledge alerts
app.put('/api/alerts/acknowledge-bulk', async (req, res) => {
  try {
    const { alertIds, acknowledgedBy } = req.body;

    if (!alertIds || !Array.isArray(alertIds) || alertIds.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'alertIds array is required'
      });
    }

    if (!acknowledgedBy) {
      return res.status(400).json({
        success: false,
        error: 'acknowledgedBy field is required'
      });
    }

    const result = await Alert.updateMany(
      { _id: { $in: alertIds }, isActive: true },
      {
        acknowledgedAt: new Date(),
        acknowledgedBy,
        isActive: false
      }
    );

    res.json({
      success: true,
      acknowledgedCount: result.modifiedCount,
      totalRequested: alertIds.length
    });
  } catch (error) {
    console.error('Error bulk acknowledging alerts:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to acknowledge alerts',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  
  // Handle specific error types
  if (error.name === 'ValidationError') {
    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: Object.values(error.errors).map(err => err.message)
    });
  }
  
  if (error.name === 'CastError') {
    return res.status(400).json({
      success: false,
      error: 'Invalid data format',
      details: error.message
    });
  }

  res.status(500).json({ 
    success: false,
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
    details: process.env.NODE_ENV === 'development' ? error.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ 
    success: false,
    error: 'Endpoint not found',
    path: req.path,
    method: req.method
  });
});

// Graceful shutdown handlers
const gracefulShutdown = async (signal) => {
  console.log(`\n${signal} received. Starting graceful shutdown...`);
  
  try {
    // Close database connection
    await mongoose.connection.close();
    console.log('Database connection closed');
    
    // Exit process
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start server
const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`Gas Monitor Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`Database: ${pprocess.env.AZURE_COSMOS_CONNECTIONSTRING || process.env.MONGODB_URI? 'Configured' : 'Not configured'}`);
  console.log(`Server ready to accept connections`);
});

module.exports = app;
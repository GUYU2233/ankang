import axios from 'axios'

const http = axios.create({ baseURL: '/api/v1', timeout: 15000 })

http.interceptors.request.use((config) => {
  const key = localStorage.getItem('ankang_api_key')
  if (key) config.headers['X-API-Key'] = key
  return config
})

http.interceptors.response.use(
  (resp) => resp.data,
  (err) => Promise.reject(err)
)

export default {
  health: () => http.get('/health'),
  dashboard: () => http.get('/dashboard/stats'),
  devices: () => http.get('/devices'),
  addDevice: (d) => http.post('/devices', d),
  updateDevice: (id, d) => http.put('/devices/' + id, d),
  deleteDevice: (id) => http.delete('/devices/' + id),
  syncDevices: () => http.post('/devices/sync'),
  localVideos: () => http.get('/streams/local-videos'),
  residents: () => http.get('/residents'),
  addResident: (r) => http.post('/residents', r),
  alerts: (level, status) => http.get('/alerts', { params: { ...(level ? { level } : {}), ...(status ? { status } : {}) } }),
  getAlert: (id) => http.get('/alerts/' + id),
  ackAlert: (id, handled) => http.post('/alerts/' + id + '/ack', null, { params: { handled: !!handled } }),
  confirmAlert: (id, payload) => http.post('/alerts/' + id + '/confirm', payload),
  verifyAlert: (id, payload) => http.post('/alerts/' + id + '/verify', payload),
  aiVerifyAlert: (id) => http.post('/alerts/' + id + '/ai-verify'),
  alertFeedback: (id) => http.get('/alerts/' + id + '/feedback'),
  alertReplay: (id) => http.get('/alerts/' + id + '/replay'),
  handleAlert: (id, payload) => http.post('/alerts/' + id + '/handle', payload),
  closeAlert: (id, payload) => http.post('/alerts/' + id + '/close', payload),
  residentDevices: (id) => http.get('/residents/' + id + '/devices'),
  residentRiskProfile: (id, days = 7) => http.get('/residents/' + id + '/risk-profile', { params: { days } }),
  riskTrend: (days = 7) => http.get('/dashboard/risk-trend', { params: { days } }),
  multimodalConfig: () => http.get('/multimodal/config'),
  saveMultimodalConfig: (c) => http.put('/multimodal/config', c),
  multimodalProviders: () => http.get('/multimodal/providers'),
  multimodalStatus: () => http.get('/multimodal/status'),
  multimodalAnalyze: (id) => http.post('/multimodal/analyze/' + id),
  multimodalResults: (params) => http.get('/multimodal/results', { params }),
  webhooks: () => http.get('/alerts/webhooks'),
  createWebhook: (d) => http.post('/alerts/webhooks', d),
  updateWebhook: (id, d) => http.put('/alerts/webhooks/' + id, d),
  deleteWebhook: (id) => http.delete('/alerts/webhooks/' + id),
  testWebhook: (id) => http.post('/alerts/webhooks/' + id + '/test'),
}
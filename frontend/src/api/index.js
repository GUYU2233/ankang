import axios from 'axios'

const http = axios.create({ baseURL: '/api/v1', timeout: 15000 })

http.interceptors.response.use(
  (resp) => resp.data,
  (err) => Promise.reject(err)
)

export default {
  health: () => http.get('/health'),
  dashboard: () => http.get('/dashboard/stats'),
  devices: () => http.get('/devices'),
  addDevice: (d) => http.post('/devices', d),
  deleteDevice: (id) => http.delete('/devices/' + id),
  syncDevices: () => http.post('/devices/sync'),
  residents: () => http.get('/residents'),
  addResident: (r) => http.post('/residents', r),
  alerts: (level) => http.get('/alerts', { params: level ? { level } : {} }),
  ackAlert: (id, handled) => http.post('/alerts/' + id + '/ack', null, { params: { handled: !!handled } })
}
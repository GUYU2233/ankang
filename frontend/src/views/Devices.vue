<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>设备接入（萤石 / 海康 / ONVIF / 模拟）</span>
        <div>
          <el-button size="small" type="primary" @click="load">刷新</el-button>
          <el-button size="small" @click="onSync">同步萤石云设备</el-button>
          <el-button size="small" type="success" @click="openAdd">登记设备</el-button>
        </div>
      </div>
    </template>
    <el-table :data="devices" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="device_name" label="名称" />
      <el-table-column prop="device_serial" label="序列号/地址" width="180" />
      <el-table-column prop="vendor" label="接入方式" width="100" />
      <el-table-column prop="scene" label="场景" width="90" />
      <el-table-column prop="status" label="状态" width="90" />
      <el-table-column label="操作" width="160">
        <template #default="s">
          <el-button size="small" type="danger" @click="onDelete(s.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="dialogVisible" title="登记设备" width="520px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="设备名称"><el-input v-model="form.device_name" /></el-form-item>
        <el-form-item label="序列号"><el-input v-model="form.device_serial" /></el-form-item>
        <el-form-item label="接入方式">
          <el-select v-model="form.vendor">
            <el-option label="模拟设备" value="sim" />
            <el-option label="萤石云" value="ezviz" />
            <el-option label="海康威视" value="hikvision" />
            <el-option label="ONVIF/RTSP" value="onvif" />
            <el-option label="RTSP直连" value="rtsp" />
          </el-select>
        </el-form-item>
        <el-form-item label="场景">
          <el-select v-model="form.scene">
            <el-option label="客厅" value="客厅" />
            <el-option label="卧室" value="卧室" />
            <el-option label="卫生间" value="卫生间" />
          </el-select>
        </el-form-item>
        <el-form-item label="RTSP/视频地址"><el-input v-model="form.access_url" placeholder="rtsp://... 或 本地mp4路径" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="onAdd">确定</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { onMounted, ref, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'

const devices = ref([])
const dialogVisible = ref(false)
const form = reactive({ device_name: '', device_serial: '', vendor: 'sim', scene: '客厅', access_url: '' })

async function load() {
  devices.value = await api.devices()
}
function openAdd() {
  Object.assign(form, { device_name: '', device_serial: '', vendor: 'sim', scene: '客厅', access_url: '' })
  dialogVisible.value = true
}
async function onAdd() {
  await api.addDevice(form)
  ElMessage.success('设备已登记')
  dialogVisible.value = false
  load()
}
async function onSync() {
  const r = await api.syncDevices()
  ElMessage.success('同步完成，新增 ' + r.added + ' 台')
  load()
}
async function onDelete(row) {
  await ElMessageBox.confirm('确认删除设备 ' + row.device_name + ' ?', '提示', { type: 'warning' })
  await api.deleteDevice(row.id)
  ElMessage.success('已删除')
  load()
}
onMounted(load)
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>